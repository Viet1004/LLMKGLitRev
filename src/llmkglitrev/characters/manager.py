"""Character management system."""

from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
import json

from .schema import ResearchCharacter


class CharacterManager:
    """
    Manage research character lifecycle.

    Handles creating, loading, updating, and deleting research characters.
    Characters are stored as JSON files in a directory structure:
    - user_defined/: User-created characters (can be modified)
    - system_default/: System-provided characters (read-only)
    """

    def __init__(self, storage_path: str = "characters/"):
        """
        Initialize character manager.

        Args:
            storage_path: Base directory for character storage
        """
        self.storage_path = Path(storage_path)
        self.user_path = self.storage_path / "user_defined"
        self.system_path = self.storage_path / "system_default"

        # Create directories if they don't exist
        self.user_path.mkdir(parents=True, exist_ok=True)
        self.system_path.mkdir(parents=True, exist_ok=True)

    def create_character(
        self,
        character: ResearchCharacter,
        is_system: bool = False
    ) -> str:
        """
        Create and save a new character.

        Args:
            character: The character to create
            is_system: If True, save as system character (read-only)

        Returns:
            The character ID
        """
        path = self.system_path if is_system else self.user_path
        file_path = path / f"{character.character_id}.json"

        # Check if character already exists
        if file_path.exists():
            raise ValueError(f"Character {character.character_id} already exists")

        with open(file_path, 'w') as f:
            f.write(character.model_dump_json(indent=2))

        return character.character_id

    def load_character(self, character_id: str) -> ResearchCharacter:
        """
        Load character by ID.

        Checks user-defined characters first, then falls back to system characters.

        Args:
            character_id: The character ID to load

        Returns:
            The loaded character

        Raises:
            ValueError: If character not found
        """
        # Try user-defined first
        user_file = self.user_path / f"{character_id}.json"
        print("User file is: ", user_file)
        if user_file.exists():
            with open(user_file) as f:
                return ResearchCharacter.model_validate_json(f.read())

        # Fall back to system default
        system_file = self.system_path / f"{character_id}.json"
        print("System file is: ", system_file)
        if system_file.exists():
            with open(system_file) as f:
                return ResearchCharacter.model_validate_json(f.read())

        raise ValueError(f"Character {character_id} not found")

    def list_characters(self, include_system: bool = True) -> List[Dict]:
        """
        List all available characters.

        Args:
            include_system: If True, include system characters

        Returns:
            List of character metadata dictionaries
        """
        characters = []

        # User-defined characters
        for file_path in self.user_path.glob("*.json"):
            try:
                with open(file_path) as f:
                    char = ResearchCharacter.model_validate_json(f.read())
                    characters.append({
                        "id": char.character_id,
                        "name": char.name,
                        "domain": char.domain,
                        "stance": char.stance,
                        "source": "user",
                        "description": char.description
                    })
            except Exception as e:
                print(f"Warning: Could not load character from {file_path}: {e}")

        # System characters
        if include_system:
            for file_path in self.system_path.glob("*.json"):
                try:
                    with open(file_path) as f:
                        char = ResearchCharacter.model_validate_json(f.read())
                        characters.append({
                            "id": char.character_id,
                            "name": char.name,
                            "domain": char.domain,
                            "stance": char.stance,
                            "source": "system",
                            "description": char.description
                        })
                except Exception as e:
                    print(f"Warning: Could not load character from {file_path}: {e}")

        return characters

    def update_character(
        self,
        character_id: str,
        updates: Dict
    ) -> ResearchCharacter:
        """
        Update existing user character.

        Only user-defined characters can be updated. System characters are read-only.

        Args:
            character_id: The character ID to update
            updates: Dictionary of field updates

        Returns:
            The updated character

        Raises:
            ValueError: If character not found or is a system character
        """
        # Check if it's a user character
        user_file = self.user_path / f"{character_id}.json"
        if not user_file.exists():
            raise ValueError(
                f"Character {character_id} not found in user-defined characters. "
                "System characters cannot be modified."
            )

        # Load existing character
        character = self.load_character(character_id)

        # Apply updates
        for key, value in updates.items():
            if hasattr(character, key):
                setattr(character, key, value)

        character.last_modified = datetime.now()

        # Save
        with open(user_file, 'w') as f:
            f.write(character.model_dump_json(indent=2))

        return character

    def delete_character(self, character_id: str):
        """
        Delete user-defined character.

        System characters cannot be deleted.

        Args:
            character_id: The character ID to delete

        Raises:
            ValueError: If character not found or is a system character
        """
        file_path = self.user_path / f"{character_id}.json"
        if file_path.exists():
            file_path.unlink()
        else:
            raise ValueError(
                f"Cannot delete system character or character not found: {character_id}"
            )

    def character_exists(self, character_id: str) -> bool:
        """
        Check if character exists.

        Args:
            character_id: The character ID to check

        Returns:
            True if character exists
        """
        user_file = self.user_path / f"{character_id}.json"
        system_file = self.system_path / f"{character_id}.json"
        return user_file.exists() or system_file.exists()

    def get_character_path(self, character_id: str) -> Optional[Path]:
        """
        Get the file path for a character.

        Args:
            character_id: The character ID

        Returns:
            Path to character file, or None if not found
        """
        user_file = self.user_path / f"{character_id}.json"
        if user_file.exists():
            return user_file

        system_file = self.system_path / f"{character_id}.json"
        if system_file.exists():
            return system_file

        return None
