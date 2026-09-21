"""Source inventory fixtures; no compiler, image or network is used."""

from pathlib import Path
import tempfile
import unittest

from source_identity import inventory


class SourceIdentityTests(unittest.TestCase):
    def test_identity_changes_with_bytes_and_is_stable_without_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "main.rs"
            source.write_text("fn main() {}")
            original = inventory(root)
            self.assertEqual(inventory(root), original)
            source.write_text("fn main() { panic!() }")
            self.assertNotEqual(inventory(root), original)

    def test_inventory_rejects_symlink_instead_of_hashing_outside_context(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "escape").symlink_to("/etc/passwd")
            with self.assertRaisesRegex(ValueError, "symlink"):
                inventory(root)


if __name__ == "__main__":
    unittest.main()
