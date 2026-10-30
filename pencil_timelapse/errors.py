"""Exception types with messages that are safe to show to end users."""


class PencilError(Exception):
    """Base class for all expected, user-facing errors."""


class InputError(PencilError):
    """An uploaded or supplied image is missing, unreadable or unsuitable."""


class ConfigError(PencilError):
    """A render setting is out of range or malformed."""


class RenderError(PencilError):
    """Frame generation or video encoding failed."""
