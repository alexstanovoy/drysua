# План разработки drysua

## 1. Назначение документа

Этот файл является основным планом разработки `drysua`. Решения по архитектуре,
обучению, производительности и взаимодействию с симулятором фиксируются здесь.

Корневой `CLAUDE.md` описывает каталог двух проектов. `drysua` является отдельным
приватным crate и зависит от sibling checkout `../bota`. Бот должен соблюдать
wire-контракт `bota-proto` и не переносить приватный код в публичный репозиторий.

Текущий статус:

- [x] Этап 1: TCP-клиент и пустая policy `Continue`.
- [x] Этап 2: синхронная in-process Arena.
- [x] Этап 3: state tracker.
- [x] Этап 4: structured action space.
- [x] Этап 5: rule-based teacher.
- [x] Этап 6: feature encoder.
- [x] Этап 7: нейронная policy.
- [x] Этап 8: behavioral cloning и DAgger.
- [x] Этап 9: PPO learner.
- [x] Этап 10: self-play league.
- [ ] Этап 11: оптимизация производительности.
- [ ] Этап 12: release gate первой обученной версии.

## 2. Цель первой версии

`drysua` играет только на Shadow Fiend:

```text
HeroId(2)
AbilityId(13) Shadowraze Near
AbilityId(14) Shadowraze Mid
AbilityId(15) Shadowraze Far
AbilityId(16) Requiem
AbilityId(17) Necromastery
AbilityId(18) Presence
```

Бот должен:

1. Играть через TCP в `Realtime` и `Lockstep`.
2. Играть in-process при обучении.
3. Работать за Radiant и Dire.
4. Управлять Shadow Fiend и courier.
5. Поддерживать весь полезный wire action space.
6. Учиться на NVIDIA CUDA, Apple Metal и CPU.
7. Не использовать данные, которых нет у обычного seat.
8. Не выигрывать за счёт дефектов симулятора.
9. Давать воспроизводимую оценку на закрытом наборе матчей.

Первый training target:

```text
1v1, MapId(0), Shadow Fiend против Shadow Fiend.
```

Код наблюдений и действий сразу должен поддерживать до 10 seats. Расширение на 5v5
не входит в первый release gate.

## 3. Основные свойства среды

Среда является частично наблюдаемой:

- enemy units исчезают в fog;
- snapshot не содержит текущий order;
- snapshot не содержит фазу attack windup/recovery;
- snapshot не содержит полный cast/channel state;
- projectile не сообщает source, target, speed и damage;
- event может ссылаться на уже удалённую сущность;
- realtime может пропускать snapshots, но не events.

Один seat отправляет не более одного order за tick. Shadow Fiend и courier делят этот
лимит. Новый body order может отменить attack recovery, channel, pending cast, item
handling и courier errand.

Поэтому `Continue` является отдельным действием. Оно не отправляет wire message и
оставляет текущий server-side order без изменений.

## 4. Зависимости

Планируемые features:

```toml
[features]
default = []
builtin = ["dep:bota-server"]
cuda = ["candle-core/cuda"]
metal = ["candle-core/metal"]
```

Планируемые зависимости:

```text
bota-proto
bota-server, optional, только для builtin Arena
candle-core 0.11
clap derive
```

Не добавляем без измеренной необходимости:

- Burn;
- tch/libtorch;
- Python runtime;
- Tokio;
- Rayon;
- Crossbeam;
- PyO3;
- отдельную tensor serialization library.

`candle-nn` на первом этапе не нужен. Adam будет реализован внутри drysua, чтобы
checkpoint содержал first moment, second moment, step и позволял точный resume.

## 5. Общая архитектура

```text
TCP Link или builtin Arena
    -> только ServerMsg конкретного seat
    -> StateTracker
    -> FeatureEncoder + ActionMask
    -> Policy
    -> StructuredAction
    -> OrderPersistence
    -> wire Order или Continue
```

Training path:

```text
CPU Arena workers
    -> bounded rollout buffers
    -> batch assembler
    -> Candle GPU learner
    -> новая policy version
    -> actors обновляют веса на границе rollout
```

Policy и reward не получают `bota_server::game::World`. Arena имеет право использовать
`World`, но наружу выдаёт только seat-specific `ServerMsg`.

## 6. Этап 1: TCP-клиент

Статус: реализован.

Контракт:

1. Подключиться с bounded timeout.
2. Включить `TCP_NODELAY`.
3. Отправить `Hello { role: Bot }`.
4. Получить `Welcome` с seat.
5. Отправить только `PickHero { HeroId(2) }`.
6. Отправить `SetReady(true)`.
7. Проверить в `MatchStart`, что seat получил Shadow Fiend.
8. Проверить совпадение mode и tick rate между `Welcome` и `MatchStart`.
9. Проверить, что snapshot имеет viewer своей команды и растущий tick.
10. В lockstep отправлять `Ack` после snapshot.
11. На первом этапе всегда выбирать `Continue`.
12. Завершаться на `MatchOver` или заданном tick limit.

Текущая команда:

```bash
cargo run --release -- \
  --addr 127.0.0.1:4455 \
  --name drysua \
  --limit 3000
```

## 7. Этап 2: in-process Arena

Статус: реализован.

Arena является синхронной. Она не создаёт поток на каждый seat.

Контракт одного матча:

- от 2 до 10 seats;
- все picks равны `HeroId(2)`;
- even slots играют Radiant;
- odd slots играют Dire;
- seed и master key строятся так же, как сервером;
- первый stream содержит `MatchStart`, затем `Snapshot(1)`;
- tick 0 никогда не показывается;
- один optional request на seat за step;
- request сначала проходит `World::validate_order`;
- rejected request возвращает исходный sequence;
- accepted requests применяются вместе;
- каждый seat получает только fog своей команды;
- события фильтруются через `EventVisibility`;
- порядок: rejection, snapshot, events, match over;
- step после `MatchOver` является ошибкой.

Обязательная проверка parity сравнивает первые два snapshot реального TCP-сервера и
builtin Arena для одного seed, map и набора picks.

## 8. Этап 3: StateTracker

Статус: реализован.

StateTracker закрывает неполноту одного snapshot.

Жёсткие лимиты первой версии:

| Структура | Лимит |
|---|---:|
| Seats | 10 |
| Current unit pointer tokens | 96 |
| Own unit memory tokens | 2 |
| Remembered non-targetable unit tokens | 32 |
| Visible units / tracked entities | 4096 |
| Observed projectiles / projectile histories | 4096 |
| Model projectile tokens | 32 |
| Loot | 16 |
| Shadow Fiend ability slots | 6 |
| Own item slots | 21 |
| Shop items | 64 |
| Point candidates | 48 |
| Events per input batch | `MAX_PAYLOAD_LEN / 2` (2,097,152) |
| Recent events | 64 |
| History | 480 ticks |

Map0 repair: [4096 bounds, hull reach, and explicit training initialization](map0-feature10-initialization.md).
The wire payload cap remains 4 MiB; selected model rows and numeric normalizers
are unchanged. Hull-inclusive attack-reach facts are corrected. The linked schemas
are now Feature v10, Model v11, PPO v19, League v20
(rules audit remains v15); this supersedes the earlier schema tuples below.

Для каждой сущности храним:

- полный `EntityId`, включая generation;
- last seen tick;
- last seen position;
- estimated velocity;
- HP и mana delta;
- last damage dealt/taken;
- last ability cast;
- death event;
- visibility state;
- estimated attack phase.

Combat history uses strictly prior events for every tracked visible or remembered
entity, including ownerless creeps and enemy units. An event batch larger than half the
wire payload byte cap is rejected before any tracker state changes. This numerical cap
is above the maximum number of `EventKind` values encodable in one valid payload.

Исторические global summaries берутся на возрастах:

```text
480, 240, 120, 60, 30, 15, 0 ticks
```

Entity ID используется только как ключ памяти и deterministic tie-break. Числовые
`idx` и `generation` не подаются модели: allocator может случайно раскрывать скрытые
события через пропуски и порядок выдачи handles.
При переполнении tracker целиком вытесняет самые старые invisible cohorts с одинаковым
last-seen tick, пока места не станет достаточно; Entity ID не выбирает запись внутри
семантически различимого cohort.

## 9. Этап 4: Structured action space

Статус: реализован.

Один плоский список действий не используется. Action является autoregressive tuple:

```text
kind
  -> controlled unit
  -> ability/item/source slot
  -> target mode
  -> entity/point/item/target slot
```

Append-only action kinds:

1. `Continue`
2. `Stop`
3. `MovePoint`
4. `FollowUnit`
5. `Hold`
6. `AttackMovePoint`
7. `AttackUnit`
8. `Cast`
9. `Use`
10. `PutPoint`
11. `PutUnit`
12. `Take`
13. `Buy`
14. `Sell`
15. `Swap`
16. `Learn`

Controlled unit:

- Shadow Fiend;
- courier.

Point candidates, максимум 48:

- 8 направлений на 3 расстояниях;
- own fountain;
- enemy fountain;
- ближайшие own/enemy towers;
- wave front;
- safe retreat point;
- predicted hero positions;
- predicted creep positions;
- nearby trees;
- building landing points;
- текущая tactical objective.

Mask строится до sampling и учитывает ownership, visibility, aim, range, mana,
cooldown, charges, inventory location, backpack mute estimate, shop range, gold, skill
points, channel и active courier errand.

Невидимые handles никогда не предлагаются как target, даже если текущая валидация
сервера ошибочно принимает такой cast.

`OrderRejected` используется только как telemetry ошибки контракта. Reason не подаётся
policy, иначе бот сможет использовать сервер как oracle скрытого состояния.

## 10. Этап 5: Rule-based teacher для Shadow Fiend

Teacher нужен для behavioral cloning и для baseline evaluation. Он не зависит от
`bota-bot`.

Приоритеты teacher:

1. Не отменять channel, attack windup и полезный persistent order.
2. Потратить skill point.
3. Купить следующий предмет.
4. Вызвать courier, забрать stash или доставить предметы.
5. Использовать heal/mana item.
6. Отступить при критическом HP или смертельном incoming damage.
7. Сделать last hit обычной атакой.
8. Сделать last hit Shadowraze, если mana trade оправдан.
9. Сделать deny.
10. Использовать Near/Mid/Far Shadowraze по дистанции и facing.
11. Использовать Requiem только при достаточном числе душ и безопасном окне.
12. Harass enemy hero без потери lane economy.
13. Атаковать structure.
14. Держать lane position.
15. Идти к текущей objective.
16. `Continue`.

Teacher отслеживает:

- attack interval и ожидаемый landing tick;
- projectile travel;
- creep HP trend;
- Shadowraze fixed distances;
- facing и время поворота;
- Necromastery souls через effects/stacks;
- enemy tower exposure;
- свой и вражеский kill range.

Каждый teacher action обязан представляться structured action space.

Release gate teacher:

```text
teacher action coverage = 100%
server rejection rate < 0.1%
```

## 11. Этап 6: Features

### 11.1. Общие правила

- Каждое значение имеет фиксированный диапазон.
- Каждое отсутствующее значение имеет presence/missing mask.
- Unknown не кодируется обычным нулём.
- Все значения проверяются на finite.
- Командные координаты канонизируются.
- Absolute side сохраняется отдельным feature для аудита асимметрии карты.
- `match_id`, seed и числовое значение `EntityId` не подаются модели.

### 11.2. Global features

Примерно 64 числа:

- normalized tick;
- pregame progress;
- wave phase;
- jungle spawn phase;
- side;
- map ID;
- seat count;
- role/lane assignment;
- K/D/A aggregates;
- XP and level advantage;
- last-hit and deny advantage;
- own gold;
- own observable asset value;
- respawn state;
- alive hero counts;
- structure HP;
- destroyed structure count;
- active order kind and age;
- ticks since last decision;
- recent damage dealt/taken.

### 11.3. Unit token

Примерно 64 числа:

- presence and relation;
- `UnitKind`;
- owner relation;
- visible/remembered;
- age since last seen;
- absolute canonical position;
- relative position;
- distance and direction;
- facing;
- radius;
- estimated velocity;
- elevation and walkability;
- HP and mana ratios;
- HP and mana deltas;
- attack damage/range/interval/speed;
- move speed;
- armor and magic resistance;
- vision and true sight;
- attacks needed to kill;
- time to reach;
- mutual attack-range flags;
- status bits;
- recent combat roles;
- estimated attack phase.

`units[96]` follows the current `ActionSpace::entity_candidates()` order exactly and is
the only unit pointer tensor. `own_units[2]` contains fixed hero/courier slots, current
or remembered. `remembered_units[32]` contains only non-targetable hidden memory, is
selected by the complete encoded semantic token and expires with tracker history.
Opaque handles may break only feature-identical ties and never enter a tensor. Unit
tokens also encode exact bounded item slot count, free slot count and capacity presence
used by `PutUnit` legality.

### 11.4. Ability tokens

Для шести fixed Shadow Fiend slots:

- ability ID;
- slot;
- level/max level;
- cooldown;
- mana cost;
- range;
- aim;
- passive/toggle;
- can level;
- local legality;
- last cast age.

Ability ID остаётся feature, хотя герой один. Это различает три Shadowraze и позволяет
проверять schema при изменении сервера.

### 11.5. Item tokens

Все 21 own slots:

- hero inventory 6;
- backpack 3;
- stash 6;
- courier 6.

Features:

- item ID;
- location and slot;
- charges;
- cooldown;
- aim and range;
- mana cost;
- attribute mode;
- for-sale state;
- estimated mute;
- value and build relation.

### 11.6. Projectiles, loot and map

Point candidate token:

- fixed prefix validity aligned with `ActionSpace::point_candidates()`;
- canonical and relative position, distance and direction;
- source category with direction, radius, unit-kind and relation parameters;
- walkable, standing-tree and allied-building semantics.

Projectile token:

- team relation;
- ability ID;
- relative position;
- facing;
- estimated velocity;
- age;
- estimated closest approach.

Loot token:

- item ID;
- charges;
- relative position;
- path distance;
- visible age.

MapContext строится один раз:

- terrain grid;
- walkable/water/elevation masks;
- opaque cells;
- spatial index trees;
- fountain and structure landmarks;
- pathfinding scratch buffers.

`MatchInfo.opaque_cells` уже содержит клетки статических деревьев на старте матча.
Этот список остаётся статическим baseline и не дополняется отдельным каналом скрытых
динамических blockers. Запись из `felled_trees` или `planted_trees` меняет policy
context и passability только тогда, когда живой allied unit находится в той же или
соседней terrain-клетке. Такое доказательство зависит только от текущих allied bodies
и геометрии: удалённая запись одного динамического дерева не может сделать видимой
запись другого. Локально доказанное срубленное статическое дерево освобождает проход,
локально доказанное посаженное дерево его закрывает, а удалённые изменения сохраняют
статический baseline.

Projectile/loot observation привязывается к точному bounded provenance: private
nonzero tracker lineage, seat, статическим входам, snapshot, его точному predecessor и
tracker history. Совпадения tick и равенства отдельно реконструированной tracker-ветки
недостаточно. Clone получает новый lineage, move сохраняет его; lineage и match ID не входят
ни в один policy tensor.
ActionSpace и readiness также сравниваются с текущими bounded входами точно; FNV hash
schema является только стабильным идентификатором descriptor, не доказательством
равенства состояния.

FeatureEncoder хранит 16 observation states. После вытеснения rollback раньше
старейшего retained snapshot возвращает точную horizon error и не меняет encoder;
rollback на самой границе остаётся точным.

Локальные readiness journals хранят восемь последних replacement requests и
эффективный вытесненный base timer. Reject поддерживается точно для retained requests;
reject уже вытесненного sequence вне normal one-outstanding-request protocol не
поддерживается. Удаление всех retained replacements восстанавливает base timer.

LocalPolicyState хранит earliest supported rollback tick. После вытеснения перехода
active order или assignment состояние на границе остаётся точным, а rollback на один
tick раньше возвращает ошибку без изменения состояния. Вытеснение decision сдвигает
horizon к tick нового decision, так как удаление более новых решений иначе вернуло бы
в feature window уже потерянное старое решение.

Ability/item payload берётся только из текущего body. Исключение — текущий hero `kit`
из scoreboard: он имеет отдельный source bit. Для отсутствующего courier remembered
payload никогда не кодируется как текущее наблюдение.

Policy получает local terrain rays и tactical distances, а не полный raster `288 x 288`.

## 12. Этап 7: модель

Первая модель использует DeepSets, а не full Transformer.

```text
unit features -> shared MLP 64 -> 128 -> 128
ability features -> shared MLP -> 64
item features -> shared MLP -> 64
projectile/loot features -> shared MLP -> 64

typed mean/max pooling
  -> heroes
  -> creeps
  -> structures
  -> neutrals
  -> couriers/wards
  -> projectiles/loot

global + history + own hero + pooled groups
  -> trunk 512 -> 256 -> 256

trunk
  -> value head
  -> action-kind head
  -> conditional heads
  -> entity pointer query
  -> point pointer query
```

Ожидаемый размер:

- 1-3 млн параметров;
- 4-12 MiB F32 weights;
- меньше 40 MiB с Adam moments;
- batch-1 CPU inference меньше 250 microseconds;
- learner batch от 2048 samples.

Первая рабочая версия полностью F32. BF16 добавляется после отдельного benchmark.

Один tensor training microbatch ограничен 64 frames. Effective learner batch
2048-8192 набирается gradient accumulation; host evaluation принимает до 8192
frames и исполняет их microbatch по 64 под одним parameter read lock.
CPU device является контрактом stage 7. Accelerator training откладывается до
отдельного benchmark и явного решения о backend portability.

## 13. Этап 8: Behavioral cloning и DAgger

Behavioral cloning запускается до reinforcement learning.

Loss:

```text
L_bc = CE(action kind) + сумма CE активных conditional heads
```

Стартовые настройки:

- Adam;
- learning rate `1e-3`;
- batch 2048-8192;
- global gradient norm clip 0.5;
- deterministic shuffle;
- отдельные held-out seeds;
- выбор лучшего training stage по gameplay evaluation.

Метрики:

- action-kind agreement;
- full structured-action agreement;
- agreement per action family;
- teacher coverage;
- rejection rate;
- score and win rate against teacher;
- result separately for Radiant and Dire.

Gate перед PPO:

```text
overall kind agreement >= 60%, full agreement >= 55%
per-side kind agreement >= 55%, full agreement >= 50%
teacher action coverage = 100%
rejection count = 0
Map 1: Weak побеждён с обеих сторон
Map 0: с обеих сторон уничтожена минимум одна enemy structure
```

DAgger:

1. Learner играет сам.
2. Собираются состояния вне teacher distribution.
3. Teacher размечает эти состояния.
4. Samples добавляются в bounded imitation pool.
5. Модель дообучается.

Статус: программный контракт реализован. Identifier-free targets строятся только из
точной пары FeatureFrame/ActionSpace; Map 1 pool ограничен 9216 samples, effective batch — 8192,
autograd microbatch — 64. Masked BC, host gradient accumulation, Adam с
`epsilon=1e-8` и clip 0.5, deterministic shuffle, side metrics и seed namespaces
используются в behavioral training. Модель выбирает один gameplay stage selector;
отдельного imitation checkpoint/early-stopping механизма нет.
Optimizer path принимает только `Train`; `Validation` и `HeldOut` доступны только для
оценки.
Один exclusive model guard покрывает весь effective update и Adam commit; ошибка epoch
восстанавливает weights, moments, counters и shuffle. Pool проверяет
map/seed/split/trajectory/tick/side identity, защищает Validation/HeldOut от FIFO eviction и
связывает trainer с точными lineage, revision, seed namespaces, action/model/
feature schemas, Shadow Fiend, typed map scope и rules-audit scope. Каждый in-memory pool имеет
неподделываемую instance identity; после DAgger mutation trainer явно принимает только
более новую revision того же instance через `rebind_pool`. Promotion принимает только
typed HeldOut evaluation с реальным teacher-attempt denominator, обе стороны, минимум
1000 rollout actions, paired gameplay и пройденные side/exploit audits.
`TeacherCoverage::collect_teacher_sample` считает `Teacher::decide` и target construction
одной попыткой, включая ошибки в denominator; successful coverage связано с точным
process-local sample instance и не создаётся публичными counter helpers.
Policy model, Adam и trainer связаны checked process-local model/optimizer lineage и
monotonic parameter revision. Raw parameter import увеличивает revision и снимает
optimizer ownership. HeldOut, rollout и каждый structural per-seed paired gameplay report
содержат одну точную `PolicyIdentity`; gate дополнительно сравнивает её с live model.
Fixed pretraining использует четыре teacher seeds и четыре DAgger seeds на Map 1,
два отдельных offline Validation seeds и три gameplay validation seeds для stage
selection, а также два ни разу не просмотренных
HeldOut seeds для финального agreement gate.
Deployment на Map 0 использует audited seat-visible Teacher, а Map 1 — model с узким
safety/objective shield. Финальный независимый seed требует две Map 1 победы и Map 0
Teacher structure progress с обеих сторон до сохранения runtime weights.
Promotion не принимает empty/one-family action corpus или zero-score gameplay; side
coverage и action distribution должны быть представлены из фактических samples.
Promotion gate остаётся data-dependent: agreement, полное coverage и gameplay gates не
считаются достигнутыми без отдельного обученного match corpus.
Для возобновляемого PPO используется файловый `TrainingArtifact` из этапа 18.
После restore promotion evidence собирается заново; process-local identity не переносит
runtime evidence между процессами и не входит в feature/model tensors.

## 14. Этап 9: PPO

Эволюция всех весов не используется. Она получает один scalar на матч и слишком плохо
масштабируется на миллионы параметров.

Начальные параметры PPO:

| Параметр | Значение |
|---|---:|
| Decision interval | 3 ticks |
| Rollout length | 2048 decisions |
| Environments | 32-128 |
| Samples/update | 8192-32768 |
| PPO epochs | 4 |
| Minibatch | 2048 или 4096 |
| Clip epsilon | 0.2 |
| Value coefficient | 0.5 |
| Entropy coefficient | 0.01 |
| Learning rate | `3e-6` |
| Adam beta1 | 0.9 |
| Adam beta2 | 0.999 |
| Adam epsilon | `1e-5` |
| Gradient clip | 0.5 |
| GAE lambda | 0.98 |
| Target KL | 0.02 |

Discount считается по прошедшим simulation ticks:

```text
discount = gamma_tick ^ ticks_since_previous_decision
```

Snapshots и events обрабатываются каждый tick. Policy обычно вызывается каждые 3 ticks.
Combat benchmark отдельно сравнивает интервалы 1, 2, 3 и 4.

Curriculum:

1. Contract and legality.
2. Shopping, learning and courier.
3. Navigation and lane arrival.
4. Last-hit and deny.
5. Shadowraze geometry and mana trade.
6. Hero combat and retreat.
7. Towers and Ancient.
8. Full match and terminal reward.

После перехода на новый stage все старые stages продолжают оцениваться.

Статус: learner и короткий реальный actor-to-learner path реализованы. Policy sampling
использует deterministic Gumbel-Max только поверх legal masks и сохраняет сумму exact
autoregressive log-probabilities, entropy, value и точную `PolicyIdentity`. Rollout
ограничен 32768 transitions и 1280 interleaved seat streams; каждый stream проверяет
монотонный decision index, elapsed ticks, terminal bootstrap и единую frozen actor
revision. Rollout другой revision отвергается до optimizer mutation.

GAE вычисляет `gamma_tick ^ elapsed_ticks`, обрывается на terminal transition и
нормализует advantages только после построения lambda returns. Learner реализует clipped
surrogate, value MSE, entropy bonus, global gradient clipping, Adam `3e-6/0.9/0.999/1e-5`,
deterministic minibatch shuffle и early stop по non-negative approximate KL. Effective
minibatch разбивается на autograd microbatches по 64, gradients суммируются на host и
применяются одной atomic model revision. При ошибке полного update выполняется rollback
coherent parameter/Adam snapshot; восстановление гарантируется только при успешном rollback.

`RewardTracker` читает только `GlobalSummary`, построенный из seat-specific protocol
state. Potential shaping имеет non-replenishing absolute-component episode budget
`100/101 < 1`, все компоненты нормализованы одной шкалой 101 и сохраняются раздельно, а
Win/Loss/Draw представлены отдельным terminal adjudication и не смешиваются с
nonterminal state. Builtin smoke держит одного learner seat против независимого teacher,
обрабатывает snapshots/events каждый tick и принимает решения раз в три ticks. Команда
`drysua train` намеренно ограничена 16 arenas, 64 decisions/environment и 10 updates:
это safety smoke, а не долгий training job.

Текущий model tensor path остаётся CPU-only согласно этапу 7. CUDA actor-learner,
double buffering и массовое использование RTX относятся к этапу 17; этап 9 не добавляет
CUDA supply-chain/build complexity и не выдаёт короткий smoke за GPU benchmark.
Server и builtin Arena завершают каждый Snapshot явным Events batch, включая пустой.
Policy принимает решение только после этой tick-complete границы, начиная с tick 1,
включая pregame; default cadence для всех policy — `1 + 3n`. Production arenas
образуют полные frozen-policy side pairs.
Current PPO contract: schema v16, hash `11450737853127354910`, rules audit v15.
Action v3 changes composite Buy legality and root/first-missing-leaf wire decoding.
Feature v8/hash `10322490384647633864` binds that action-dependent legality; model
v8/hash `3097714014199697774` binds the new action/feature contracts without changing
tensor shapes or parameter order. See [PPO v16 migration](ppo-v16-migration.md).
PPO value regression detaches the value-head input and trains only that head, not shared
actor features; this isolation does not alter BC, public training forward, parameter order
or architecture. KL is checked before and after the candidate optimizer step. Post-step KL
is sample-weighted over the complete effective minibatch against the rollout policy.
Overshoot or candidate-evaluation error restores exact parameters, Adam moments/step and
policy revision under the exclusive parameter lock when rollback succeeds. If the backend
fails during rollback import, the error retains both causes; atomic restoration is not
guaranteed. Applied reports contain post-step KL;
post-step rejection reports contain candidate KL. This is not full-distribution KL or an
anchor-wide multi-update drift guarantee. Permanent critic isolation, LR `3e-6`, and reward
normalization are conservative design/tuning decisions, not evidence that standard shared
actor/critic gradients are incorrect or that learned strength improved. GAE is unchanged;
the v15 reward contract is below. See [PPO v15 migration](ppo-v15-migration.md).

## 15. Reward

Terminal reward:

```text
win  = +1
loss = -1
draw = 0
truncation without terminal adjudication = bootstrap, not a terminal draw
```

Episode shaping budget ограничен `100/101` для суммы абсолютных emitted components;
смена знака и взаимная компенсация компонентов не восполняют budget. Expenditure хранится
в f64, emitted rewards — f32 (tolerance `1e-6` для rounding). Все shaping components
масштабируются пропорционально оставшемуся budget. Terminal dominance относится к
undiscounted episode budget, не к произвольно далёким discounted outcomes. Clipped
potential shaping не даёт общей гарантии policy invariance.

Текущие компоненты v15:

- XP advantage: `.02/101`;
- kills/deaths combat advantage: `2/101`;
- structure destruction advantage: `5/101`;
- cash wealth, last hits и denies: zero reward; покупки не штрафуются за расход cash.

Остальные перечислявшиеся ранее сигналы не являются текущими reward components.
Потенциал рассчитывается как:

```text
gamma * Phi(next_state) - Phi(current_state)
```

При любом terminal outcome `Phi(next_state) = 0`; финальный snapshot не влияет на
terminal shaping до применения budget. Truncation сохраняет discounted next potential.

Reward считается только из `MatchInfo`, seat-specific `WorldView`, visible `Events` и
`MatchOver`. Reward code не читает полный `World`.

Каждый компонент логируется отдельно. Общего числа без breakdown недостаточно для
поиска reward hacking.

## 16. Этап 10: self-play league

Один frozen opponent не используется как единственный соперник.

Начальное распределение opponents:

- 30% current policy mirror;
- 25% последний accepted checkpoint;
- 25% historical league;
- 15% teacher;
- 5% weak/random baseline.

League bounded, начальный лимит 32 policies. При переполнении сохраняются:

- strongest;
- recent;
- старые полезные anchors;
- policies с отличающимся cross-play profile.

Каждый evaluation seed играется с перестановкой сторон. Promotion checkpoint происходит
только после held-out gate, а не по training reward.

Статус: реализован bounded league на 32 immutable policy snapshots с распределением
opponents 30/25/25/15/5. Frozen model opponents материализуются отдельно от learner и
не меняются внутри rollout; actor RNG использует отдельные domain-separated streams и не
передаёт simulator seed в model inputs. Retention сохраняет anchor, accepted, strongest,
четыре recent policies и отличающийся трёхосевой cross-play profile; минимальная capacity
9 гарантирует evictable slot.

`drysua league` выполняет bounded self-play PPO update, paired held-out evaluation,
teacher/accepted/historical cross-play и отдельный paired exploit-regression namespace.
Truncated matches считаются Draw и не превращаются в победу по training reward или
промежуточным public statistics. Promotion требует минимум 20 disjoint paired seeds,
1000 candidate actions, rejection rate ниже 0.1%, отсутствия regression на каждой стороне
и opaque exploit audit, привязанный к candidate и accepted fingerprints. Stage-ten
Current league contract: schema v17, hash `14981756892451872554`, rules audit v15.

## 17. GPU и actor-learner pipeline

```text
CPU environment workers
    -> rollout buffer A/B
    -> batch assembler
    -> one CUDA/Metal learner
    -> policy version update
```

Правила:

- simulation остаётся на CPU;
- worker count bounded;
- каждый worker владеет несколькими arenas;
- queues bounded;
- policy меняется только на rollout boundary;
- sample хранит old log-probability, value and policy version;
- rollout старше одной policy version отклоняется;
- CPU rollout и GPU update перекрываются double buffering.

Rollout storage является ragged:

```text
sample headers
unit arena + offsets
ability arena + offsets
item arena + offsets
point arena + offsets
bit-packed masks
actions
old log probabilities
values
rewards
done flags
policy versions
```

Padding выполняется только при сборке minibatch.

Статус: реализованы bounded worker endpoints и два общих владельца rollout buffer,
атомарная выдача immutable CPU policy lease вместе с generation, отклонение rollout с lag
больше одной generation и отдельный trusted путь learner update для допустимого lag. CUDA и
Metal выбираются явно через `PolicyDevice`; параметры, forward, loss и backward находятся на
выбранном backend. Rollout хранит sparse token rows в typed arenas с проверяемыми offsets,
bit-packed legal masks и разворачивает fixed padding только для текущего minibatch. Model
schema v8, hash `3097714014199697774`; PPO schema v16, hash `11450737853127354910`.

## 18. Checkpoints

Runtime artifact:

```text
drysua.weights.safetensors
```

Training artifact:

```text
checkpoint.safetensors
checkpoint.meta
```

Training checkpoint содержит:

- policy and value weights;
- Adam first and second moments;
- optimizer step;
- scheduler state;
- global update;
- policy version;
- RNG states;
- curriculum stage;
- rollout counters;
- best evaluation;
- league references.

Manifest содержит:

- schema version;
- observation schema hash;
- action schema hash;
- git commit;
- enabled features;
- map and hero scope;
- run seed;
- device and dtype;
- batch size;
- command line;
- tensor file hash;
- simulator commit and rules audit version.

Load является strict: точные names, shapes, dtype, finite values и schema. Partial load с
random weights запрещён. Save выполняется через temporary file и atomic rename.

После изменения правил симулятора старые datasets и checkpoints помечаются несовместимыми,
пока отдельная evaluation не докажет обратное.

Статус: реализованы deployment-only `drysua.weights.safetensors` и полный training artifact
`checkpoint.safetensors` + `checkpoint.meta`. Manifest хранит exact action/feature/model/PPO
schemas, git и simulator commits, Cargo features, hero/map scope, run seed, backend, batch,
command line, rules audit, scheduler/curriculum/rollout counters, best evaluation, bounded RNG
states и league references. Training tensors содержат policy/value parameters и оба Adam
moment vectors; optimizer step, trainer update и shuffle state находятся в manifest. Load
проверяет bounded file sizes, SHA-256, exact tensor names/shapes/F32, finite/nonnegative state и
schema до атомарной установки model+optimizer ownership. Save сохраняет immutable SHA-256
generation, canonical tensor copy и recoverable manifest последним; каждый файл и directory
fsync-ится. Двухфайловая копия остаётся independently loadable через hash-checked canonical
fallback. Runtime weights additionally bind PPO schema/version and rules audit metadata.
Training resume and runtime reads are strictly current-schema. PPO v13/v14/v15 training
manifests are rejected, including under `--migrate-provenance`. The former exact v13 runtime
exception is removed: action v2 semantics are incompatible with action v3 even though tensor
shapes match. Historical BC anchor `e63f0bdb478ebb8b` and probe weights cannot initialize this
build. Missing/extra metadata fields, wrong hashes and mixed tuples reject before model
mutation; names, shapes, F32 and finite checks remain mandatory. Historical tag artifacts
stay immutable and run only with their matching historical binaries/source trees. Do not
rewrite metadata or bypass validation. The checkpoint wire layout itself is unchanged.
Checkpoint schema v2, hash `4581258024746721724`.

`drysua evaluate` загружает runtime artifact и всегда запускает фиксированную матрицу из обеих
карт, teacher/weak baseline и обеих сторон на одинаковых held-out seeds. Timeout отделён от draw;
hard gate отклоняет all-timeout, no-order, server-rejected и >=95% single-action collapse runs.
Эта итерационная suite не заменяет отдельные sealed promotion seeds.

## 19. Производительность

Обязательные крупные улучшения:

| Оптимизация | Ожидаемый результат |
|---|---:|
| In-process Arena | кратный прирост, по старому измерению до порядка 20x |
| Release simulation | крупный прирост ticks/s |
| Decision раз в 3 ticks | до 3x меньше inference и samples |
| Order deduplication | меньше wire work и animation cancels |
| Fixed CPU worker pool | масштабирование по cores |
| GPU learner batches | кратный прирост training math |
| Actor/learner double buffer | до примерно 2x при балансе стадий |
| Ragged rollout storage | меньше RAM и transfer |
| Preallocated scratch | отсутствие hot-loop allocations |
| DeepSets | линейная стоимость по entities |

Оптимизация включается по умолчанию, если даёт не менее 10% end-to-end throughput и не
ухудшает held-out gameplay.

Отдельно измеряются:

- BF16;
- GPU actor inference;
- custom CPU inference kernel;
- decision intervals;
- compressed offline storage.

Статус профилирования (release, 16 demo arenas x 2000 ticks, counting allocator): baseline
Arena `175632 ticks/s`, `96.57 allocations/tick`, `17408 bytes/tick`; после reuse одного
bounded entity snapshot во всех частых simulator systems — `194524 ticks/s` (+10.8%),
`53.17 allocations/tick` (-44.9%), `14084 bytes/tick`. Чистый `World::advance` вырос с
`228126` до `264440 ticks/s` (+15.9%). Изменение включено по умолчанию, так как прошло 10%
threshold и полную deterministic simulator suite. Profiling harness имеет безопасные defaults
2 arenas x 500 ticks, максимум 16 arenas, 20000 ticks и 2 updates.

Thin LTO дал только +6.9% поверх оптимизированного simulator и поэтому не включён по умолчанию.
Workload-specific PGO дал `223996 Arena ticks/s` (+15.1% поверх обычного release); bounded
`scripts/pgo-build.sh` воспроизводит instrument/merge/build отдельно от portable release.
NVFP4 probe на RTX 5090/CUDA 13.3 завершился `CUDA_ERROR_NOT_FOUND` для Candle F4 kernel, поэтому
неподдерживаемый precision не включён; learner остаётся strict F32. На маленьком безопасном batch
128 transitions CUDA не быстрее CPU (`116` против `119 samples/s`), хотя host allocations bytes
сократились примерно вдвое; больший batch не запускался после resource-safety stop.

Reward v3 больше не оптимизирует LH/denies: их breakdown остаётся diagnostic и всегда равен нулю.
Observable own gold имеет weight 0.01, allied-minus-enemy public XP — 0.02. Enemy gold отсутствует
в seat protocol и намеренно не добавлен, чтобы не нарушать anti-cheat boundary. Demo MapId(1)
теперь заканчивается при первом падении tower или второй суммарной смерти стороны. Из 37896 строк
`bota-server` 21123 приходятся на generated terrain/tree tables и tests; исполняемый handwritten
server — около 16773 строк, поэтому широкое сокращение без найденного профилем дефекта отклонено.

## 20. Угроза читерства через симулятор

### 20.1. Определение

Читерством считается policy, которая получает преимущество из:

- данных, недоступных обычному seat;
- бага validation;
- утечки скрытого состояния через protocol metadata;
- разницы TCP и builtin paths;
- бага terminal condition;
- явно незавершённой механики;
- сильной случайной асимметрии сторон;
- reward/statistics bug;
- знания training seeds или RNG state.

Сильная игра в рамках явно реализованной и одинаковой для всех механики читерством не
считается. Граница определяется simulator contract и отдельным решением команды.

### 20.2. Жёсткая изоляция policy

Следующие правила действуют даже до исправления сервера:

1. Policy получает только seat-specific messages.
2. Reward получает только тот же stream и terminal result.
3. `World`, RNG и server components не передаются model/teacher/reward.
4. `match_id` и seed не входят в features.
5. Числовые EntityId не входят в features.
6. Invisible handles не входят в target candidates.
7. Rejection reason не входит в observation.
8. Fogless replay не используется как policy dataset.
9. Spectator snapshots не используются для imitation.
10. Builtin/TCP parity является обязательным regression test.
11. Evaluation всегда side-paired.
12. Training и evaluation seeds не пересекаются.

### 20.3. Известные риски и дефекты текущего симулятора

Ниже перечислены известные места, которые policy может научиться эксплуатировать. Перед
обучением соответствующей механике нужен simulator test, решение о contract и при
необходимости contribution в основной репозиторий.

Список является обязательным стартовым реестром, а не закрытым перечнем. Любая новая
аномалия из training replay добавляется сюда до продолжения обучения.

#### A. Утечки скрытого состояния

| Проблема | Возможный exploit | Временная защита drysua | Нужный contribution |
|---|---|---|---|
| `match_id` фактически равен CLI seed, а master key строится из его байтов | Предсказать RNG или запомнить trajectory | Не подавать match_id/seed модели | Отделить публичный match ID от секретного RNG seed |
| Enemy `PlayerView.unit` доступен через fog | Получить актуальный handle скрытого hero | Не создавать target из невидимого unit | Пересмотреть wire field или validation |
| Cast по unit не проверяет visibility | Идти к точной позиции enemy в fog | Только currently visible cast targets | Проверять visibility при validation и execution |
| Use по unit не проверяет visibility, alliance и range полностью | Пробовать hidden targets и получать oracle | Консервативный local mask | Исправить item validation |
| `felled_trees` и `planted_trees` глобальны | Узнавать действия enemy в fog | Не использовать удалённые tree changes как enemy signal | Fog-filter tree changes либо признать их публичными |
| Entity IDs выдаются глобальным allocator | Пропуски ID могут раскрывать hidden spawn/death | Не подавать numeric ID модели | Проверить возможность per-view opaque IDs |
| Enemy XP, level, K/D/A, LH and denies глобальны | Получать стратегическую информацию без vision | Отдельный audit feature flag | Явно определить scoreboard contract |
| Fogless server replay | Обучиться на невозможных наблюдениях | Не использовать replay напрямую | Добавить seat-projected replay/export |
| Builtin crate имеет доступ к `World` | Случайно построить privileged critic/reward | Type boundary только через `ServerMsg` | Вынести generic seat-only Arena API в server при необходимости |
| Ground loot не сообщает ownership | Проверять ownership действиями и читать rejection | Поднимать только предмет с известным происхождением | Добавить доступную своей стороне ownership metadata |

#### B. Validation и execution расходятся

| Проблема | Возможный exploit | Временная защита | Нужный contribution |
|---|---|---|---|
| Out-of-range cast принимается и автоматически ведёт hero к target | Hidden-target pathing или неожиданный macro action | Только visible target; явно моделировать persistent cast | Уточнить и протестировать contract |
| Out-of-range item use может быть принят, но ничего не сделать | Использовать accepted/no-op как oracle | Проверять range локально | Возвращать rejection либо authoritative execution result |
| Нет подтверждения accepted-and-executed | Нельзя отличить успешное действие от silent no-op | Отслеживать наблюдаемый эффект, не reward сам order | Добавить execution result/event |
| Item backpack mute/shared cooldown не полностью видны | Провоцировать rejection и извлекать hidden timer | Локально отслеживать известные timers | Добавить own-seat state на wire |
| Новый повторный order отменяет recovery/channel | Policy может случайно или специально ломать timing | `Continue` и OrderPersistence | Добавить server-side idempotence только если это intended |
| Sequence не проверяется на монотонность | Возможные protocol edge exploits | Всегда монотонный checked sequence | Добавить validation, если sequence станет semantic |

#### C. Terminal condition и структуры

| Проблема | Возможный exploit | Временная защита | Нужный contribution |
|---|---|---|---|
| Нет общего time limit/draw | Policy может избегать поражения бесконечно | Bounded training limit с честным adjudication | Добавить server rules для timeout/draw |

#### D. Экономика и статистика

| Проблема | Возможный exploit | Временная защита | Нужный contribution |
|---|---|---|---|
| Assists не начисляются | Командная помощь не имеет статистической цены | Не использовать assists reward | Реализовать assist attribution |
| `last_hits` включает neutrals и towers | Reward путает разные действия | Классифицировать через cached UnitKind/events | Разделить counters |
| `net_worth` означает cumulative earned gold | Policy/report может оптимизировать неверную метрику | Считать observable assets самостоятельно | Исправить имя/семантику или добавить новую metric |
| `hero_damage` и `structure_damage` всегда нули | Evaluation скрывает реальную силу | Считать visible events и structure HP delta | Накапливать authoritative stats |
| `Healed` не производится | Нельзя оценить healing по event | Считать HP delta с оговорками | Производить authoritative event |

#### E. Shadow Fiend и combat balance

| Проблема | Возможный exploit | Временная защита | Нужный contribution |
|---|---|---|---|
| Generic cast point отсутствует | Shadowraze/Requiem могут быть слишком быстрыми | Teacher/model используют фактические правила | Добавить cast points, если они входят в intended game |
| Requiem не расходует souls | Повторное использование может быть слишком выгодным | Измерить cooldown/soul economy | Явно решить balance contract |
| Ability/item event coverage неполна | Нельзя авторитетно связать outcome с cast | Reward по эффекту, не по факту cast | Исправить events |

#### F. Стороны и карта

| Проблема | Возможный exploit | Временная защита | Нужный contribution |
|---|---|---|---|
| Map 0 не является точным зеркалом | Одна сторона может иметь устойчивое преимущество | Каждый seed играть с swap сторон | Исправлять только доказанную нечестную геометрию |
| Safe/hard lanes различаются | Role distribution может быть несбалансирован | Report per side/role | Отдельный balance audit |
| Structures всегда видимы | Это может быть intended или лишняя информация | Считать публичной до решения | Зафиксировать contract |
| Unknown MapId откатывается к map 0 | Info и фактическая карта могут расходиться | Принимать только известные MapId | Сервер должен reject unknown map |
| Tick rate CLI меняет wall clock, но не game balance constants | Неверная интерпретация времени внешним bot code | Все gameplay timers считать в ticks | Валидировать supported tick rate или уточнить docs |

#### G. Training dynamics, усиливающие дефекты симулятора

| Проблема | Возможный exploit | Временная защита | Нужное действие |
|---|---|---|---|
| Одна policy играет за обе стороны и получает усреднённый score | Сговориться на мирной игре, где никто не мешает farm | PPO advantage по seat, margins и opponent pool | Cross-play audit против независимых checkpoints |
| Training timeout не равен игровому terminal result | Избегать поражения до конца rollout | Явное adjudication, отдельная метрика timeout | Добавить server time limit/draw contract |
| Reward использует snapshot до events того же tick | Получать credit на неверное действие | Закрывать tick только после полного message set | Parity test на snapshot/events ordering |
| Fixed curriculum seeds повторяются | Запомнить match trajectory без прямого seed feature | Rotating train seeds и sealed evaluation | Seed namespace tests |
| Teacher знает server constants, которых нет на wire | Train/serve drift после изменения balance | Версионированная semantic table | Добавить нужные static rules в wire либо обновлять schema вместе с server |

### 20.4. Автоматический exploit audit

Перед каждым новым curriculum stage запускаются специальные проверки.

#### Seed leakage audit

- Обучить маленький probe с match_id/EntityId features и без них.
- Если leak features дают значимый held-out прирост, training блокируется.
- Evaluation seeds генерируются из отдельного namespace.
- Один seed никогда не используется одновременно для selection и reporting.

#### Side advantage audit

- Каждый matchup играется за обе стороны.
- Считаются Wilson confidence intervals win rate по стороне.
- Сравниваются game length, XP, LH, deaths, tower timing.
- Устойчивый side gap воспроизводится минимальным server test.

#### Visibility audit

- Policy action target обязан присутствовать в текущем seat view.
- Feature provenance test связывает каждый model field с конкретным wire field или
  bounded local history.
- Удаление hidden enemy из view не должно менять features, кроме разрешённой памяти.
- Spectator/full view никогда не попадает в rollout buffer.

#### Validation fuzzing

- Генерировать structured actions на границах range, mana, cooldown, slot and visibility.
- Сравнивать local mask, `validate_order`, rejection и наблюдаемый execution.
- Любое `mask legal + server reject` является contract bug.
- Любое `accepted + no effect`, которое не является persistent action, расследуется.

#### Rules completeness audit

- Проверять Ancient gating.
- Проверять tower path opening.
- Проверять hero bounty, XP, gold loss and assists.
- Проверять Shadowraze distances and shared levels.
- Проверять Requiem soul behavior.
- Проверять projectile/elevation rules.
- Проверять item cooldown/mute state.

#### Degenerate-policy audit

Запускаются policies, которые специально пытаются:

- сразу идти к Ancient;
- бесконечно избегать боя;
- стоять у fountain;
- спамить hidden handles;
- спамить invalid items;
- отменять и повторять orders каждый tick;
- hoard gold;
- кормиться смертями без gold loss;
- использовать courier как oracle;
- запоминать seed;
- получать reward без прогресса к победе.

Если degenerate policy выигрывает или получает высокий curriculum score, обучение
останавливается до исправления reward либо симулятора.

## 21. Процесс contribution в симулятор

Когда обнаружен exploit или недоработка:

1. Остановить promotion затронутых checkpoints.
2. Сохранить seed, стороны, orders, snapshots и events.
3. Сделать минимальный deterministic reproduction.
4. Классифицировать проблему:
   - information leak;
   - validation mismatch;
   - simulation bug;
   - side imbalance;
   - incomplete balance;
   - intended mechanic;
   - reward bug только в drysua.
5. Добавить failing regression test в соответствующий simulator subsystem.
6. Исправить simulator, а не маскировать существенный дефект только в policy.
7. Проверить debug/release determinism и TCP/builtin parity.
8. Увеличить `rules audit version` drysua.
9. Пометить старые rollouts несовместимыми.
10. Повторить teacher, BC, side and exploit evaluations.

Если поведение признано intended, решение документируется в simulator design и
добавляется positive contract test. Неясная механика не считается стабильной только
потому, что policy научилась её использовать.

## 22. Evaluation

Минимальный report:

- wins/losses/draws;
- result per side;
- confidence interval;
- game length;
- level and XP;
- last hits and denies;
- kills and deaths;
- tower timing;
- observable net worth;
- action distribution;
- Continue ratio;
- rejection rate and reasons;
- accepted-but-no-observed-effect count;
- reward component breakdown;
- exploit audit results;
- simulator commit and rules audit version.

Наборы seeds:

- training rotating seeds;
- validation fixed but не используемый optimizer;
- promotion held-out;
- final report sealed до release candidate;
- exploit regression seeds.

Promotion требует улучшения не только общего score, но и отсутствия regressions в
legality, side gap и exploit audits.

## 23. Тестирование drysua

### Unit

- feature ranges and finite values;
- missing masks;
- canonical coordinates;
- entity generation handling;
- bounded overflow ordering;
- history by ticks;
- action encode/decode;
- conditional masks;
- Continue sends nothing;
- repeated order suppression;
- Shadowraze target geometry;
- reward positive/negative boundaries;
- terminal dominates shaping;
- model shapes and finite gradients;
- Adam reference update;
- strict checkpoint load.

### Integration

- TCP/builtin trajectory parity;
- first snapshot tick 1;
- lockstep waits for all seats;
- fog and event visibility;
- side canonicalization;
- every action family accepted;
- channel and recovery persistence;
- courier delivery;
- same seed repeats;
- different seeds remain separated;
- no model feature has privileged provenance.

### Training

- tiny BC batch reduces loss;
- PPO synthetic bandit improves;
- policy ratio is one before update;
- clipped objective boundaries;
- rollout policy version checks;
- CPU checkpoint resume repeats next update;
- train/evaluation seed sets do not intersect;
- league remains bounded.

### Performance

- ticks/s one arena;
- ticks/s worker pool;
- decisions/s;
- learner samples/s;
- CPU batch-1 inference;
- CUDA/Metal batch throughput;
- transfer time;
- full update wall time;
- peak RAM and GPU memory.

## 24. CLI roadmap

```text
drysua play
drysua teacher
drysua imitate
drysua train
drysua league
drysua evaluate
drysua duel
drysua benchmark
drysua audit
drysua inspect
```

Примеры будущих команд:

```bash
cargo run --release --features builtin,cuda -- \
  imitate --matches 10000 --device cuda
```

```bash
cargo run --release --features builtin,cuda -- \
  train --workers 16 --environments 64 --rollout 256 --minibatch 4096
```

```bash
cargo run --release --features builtin -- \
  audit --seeds 200 --paired-sides
```

## 25. Definition of done первой обученной версии

Первая версия готова, когда одновременно выполнено:

1. Teacher action coverage 100%.
2. BC full-action agreement не ниже 95%.
3. Server rejection rate ниже 0.1%.
4. PPO checkpoint лучше teacher на held-out paired matches.
5. Нет statistically significant unexplained side advantage.
6. Нет известного simulator exploit, который policy использует для победы.
7. Ancient, hero bounty и другие blocking full-game mechanics исправлены или явно
   исключены из release ruleset.
8. TCP и builtin paths дают одинаковые seat trajectories.
9. CPU runtime загружает strict F32 weights и играет полный матч.
10. CUDA training может точно продолжиться из checkpoint.
11. Debug/release tests проходят.
12. Performance report сохранён рядом с release checkpoint.

## 26. Отложенные решения

Не входят в первую версию без измеренного требования:

- recurrent policy;
- full Transformer;
- privileged centralized critic;
- Python/PyTorch trainer;
- multi-GPU DDP;
- 5v5 curriculum;
- direct coordinate regression;
- обучение из fogless server replay;
- evolution основной нейронной сети.

Каждое из этих решений может быть добавлено после профилирования или доказанного
ограничения текущей архитектуры.
