from collections.abc import Iterator
from pathlib import Path
import random
import sys
from typing import ClassVar

from loguru import logger
from rich.text import Text
from textual import events
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen, Screen
from textual.widgets import Digits, Footer, Header, Label, Log, RichLog, Static

from glados.engine import Glados, GladosConfig
from glados.glados_ui.text_resources import aperture, help_text, login_text, recipe

# Custom Widgets


class Printer(RichLog):
    """A subclass of textual's RichLog which captures and displays all print calls."""

    def on_mount(self) -> None:
        self.wrap = True
        self.markup = True
        self.begin_capture_print()

    def on_print(self, event: events.Print) -> None:
        # Only show SUCCESS-level messages in the UI log
        if (text := event.text) != "\n" and "SUCCESS" in text:
            self.write(text.rstrip())


class ScrollingBlocks(Log):
    """A widget for displaying random scrolling blocks."""

    BLOCKS = "⚊⚌☰𝌆䷀"
    DEFAULT_CSS = """
    ScrollingBlocks {
        scrollbar_size: 0 0;
        overflow-x: hidden;
    }"""

    def _animate_blocks(self) -> None:
        # Create a string of blocks of the right length, allowing
        # for border and padding
        """
        Generates and writes a line of random block characters to the log.

        This method creates a string of random block characters with a length adjusted
        to fit the current widget width, accounting for border and padding. Each block
        is randomly selected from the predefined `BLOCKS` attribute.

        The generated line is written to the log using `write_line()`, creating a
        visually dynamic scrolling effect of random block characters.

        Parameters:
            None

        Returns:
            None
        """
        random_blocks = " ".join(random.choice(self.BLOCKS) for _ in range(self.size.width - 8))
        self.write_line(f"{random_blocks}")

    def on_show(self) -> None:
        """
        Set up an interval timer to periodically animate scrolling blocks.

        This method is called when the widget becomes visible, initiating a recurring animation
        that calls the `_animate_blocks` method at a fixed time interval of 0.18 seconds.

        The interval timer ensures continuous block animation while the widget is displayed.
        """
        self.set_interval(0.18, self._animate_blocks)


class Typewriter(Static):
    """A widget which displays text a character at a time."""

    def __init__(
        self,
        text: str = "_",
        id: str | None = "",
        speed: float = 0.01,  # time between each character
        repeat: bool = False,  # whether to start again at the end
        *args: str,
        **kwargs: str,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._text = text
        self.__id = id
        self._speed = speed
        self._repeat = repeat
        self._blink_counter = 0

    def compose(self) -> ComposeResult:
        self._static = Static(markup=True)
        self._vertical_scroll = VerticalScroll(self._static, id=self.__id)
        yield self._vertical_scroll

    def _is_safe_position(self, text: str, pos: int) -> bool:
        """Check if slicing the text at pos results in balanced markup."""
        stack = []
        i = 0
        while i < pos:
            if text[i] == '[':
                if i + 1 < len(text) and text[i + 1] == '/':
                    # Closing tag
                    j = i + 2
                    tag = ''
                    while j < len(text) and text[j] != ']':
                        tag += text[j]
                        j += 1
                    if stack and stack[-1] == tag:
                        stack.pop()
                    i = j
                else:
                    # Opening tag
                    j = i + 1
                    tag = ''
                    while j < len(text) and text[j] != ']':
                        tag += text[j]
                        j += 1
                    stack.append(tag)
                    i = j
            else:
                i += 1
        return len(stack) == 0

    def _get_iterator(self) -> Iterator[str]:
        safe_positions = [i for i in range(len(self._text) + 1) if self._is_safe_position(self._text, i)]
        return (self._text[:i] for i in safe_positions)

    def on_mount(self) -> None:
        self._iter_text = self._get_iterator()
        self.set_interval(self._speed, self._display_next_char)

    def _display_next_char(self) -> None:
        """Get and display the next character."""
        try:
            if not self._vertical_scroll.is_vertical_scroll_end:
                self._vertical_scroll.scroll_down()
            base_text = next(self._iter_text)
            cursor = "▃" if (self._blink_counter // 5) % 2 == 0 else " "
            self._static.update(base_text + cursor)
            self._blink_counter += 1
        except StopIteration:
            if self._repeat:
                self._iter_text = self._get_iterator()
                self._blink_counter = 0


# Screens


class SplashScreen(Screen[None]):
    """Splash screen shown on startup."""

    # with open(Path("src/glados/glados_ui/images/converted.ansi"), encoding="utf-8") as f:
    #     SPLASH_ANSI = Text.from_ansi(f.read(), no_wrap=True, end="")

    def compose(self) -> ComposeResult:
        """
        Compose the layout for the splash screen.

        This method defines the visual composition of the SplashScreen, creating a container
        with a logo, a banner, and a typewriter-style login text.

        Returns:
            ComposeResult: A generator yielding the screen's UI components, including:
                - A container with a static ANSI logo
                - A label displaying the aperture text
                - A typewriter-animated login text with a slow character reveal speed
        """
        with Container(id="splash_logo_container"):
            # yield Static(self.SPLASH_ANSI, id="splash_logo")
            yield Label(aperture, id="banner")
        yield Typewriter(login_text, id="login_text", speed=0.0075)

    def on_mount(self) -> None:
        """
        Automatically scroll the widget to its bottom at regular intervals.

        This method sets up a periodic timer to ensure the widget always displays
        the most recent content by scrolling to the end. The scrolling occurs
        every 0.5 seconds, providing a smooth and continuous view of the latest information.

        Args:
            None

        Returns:
            None
        """
        self.set_interval(0.5, self.scroll_end)

    def on_key(self, event: events.Key) -> None:
        """
        Handle key press events on the splash screen.

        This method is triggered when a key is pressed during the splash screen display.
        If the 'q' key is pressed, it triggers the application quit action.
        Regardless of the key pressed, it dismisses the current splash screen
        and starts the main GLADOS application.

        Args:
            event (events.Key): The key event that was triggered.
        """
        # fire her up.....

    def on_key(self, event: events.Key) -> None:
        if event.key == "q":
            self.app.action_quit()  # Use self.app instead of global app
        self.dismiss()
        self.app.start_glados()  # Use self.app instead of global app


class HelpScreen(ModalScreen[None]):
    """The help screen. Possibly not that helpful."""

    BINDINGS: ClassVar[list[Binding | tuple[str, str] | tuple[str, str, str]]] = [
        ("escape", "app.pop_screen", "Close screen")
    ]

    TITLE = "Help"

    def compose(self) -> ComposeResult:
        """
        Compose the help screen's layout by creating a container with a typewriter widget.

        This method generates the visual composition of the help screen, wrapping the help text
        in a Typewriter widget for an animated text display within a Container.

        Returns:
            ComposeResult: A generator yielding the composed help screen container with animated text.
        """
        yield Container(Typewriter(help_text, id="help_text"), id="help_dialog")

    def on_mount(self) -> None:
        dialog = self.query_one("#help_dialog")
        dialog.border_title = self.TITLE
        dialog.border_subtitle = "[blink]Press Esc key to continue[/]"


# The App


class GladosUI(App[None]):
    """The main app class for the GlaDOS ui."""

    BINDINGS: ClassVar[list[Binding | tuple[str, str] | tuple[str, str, str]]] = [
        Binding(key="q", action="quit", description="Quit"),
        Binding(
            key="question_mark",
            action="help",
            description="Help",
            key_display="?",
        ),
        Binding(key="space", action="start_listening", description="Start listening (manual)", key_display="Space"),
    ]
    CSS_PATH = "glados_ui/glados.tcss"

    ENABLE_COMMAND_PALETTE = False

    TITLE = "Soni Assistant v 2.36"

    SUB_TITLE = "(c) 2024 SoICT Innovation Club."

    with open(Path("src/glados/glados_ui/images/converted.ansi"), encoding="utf-8") as f:
        LOGO_ANSI = Text.from_ansi(f.read(), no_wrap=True, end="")

    def compose(self) -> ComposeResult:
        """
        Compose the user interface layout for the GladosUI application.

        This method generates the primary UI components, including a header, body with log and utility areas,
        a footer, and additional decorative blocks. The layout is structured to display:
        - A header with a clock
        - A body containing:
          - A log area (Printer widget)
          - A utility area with a typewriter displaying a recipe
        - A footer
        - Additional decorative elements like scrolling blocks, text digits, and a logo

        Returns:
            ComposeResult: A generator yielding Textual UI components for rendering
        """
        # It would be nice to have the date in the header, but see:
        # https://github.com/Textualize/textual/issues/4666
        yield Header(show_clock=True)

        with Container(id="body"):
            with Horizontal():
                yield (Printer(id="log_area"))
                with Container(id="utility_area"):
                    typewriter = Typewriter(recipe, id="recipe", speed=0.01, repeat=True)
                    yield typewriter

        yield Footer()

        # Blocks are displayed in a different layer, and out of the normal flow
        with Container(id="block_container", classes="fadeable"):
            yield ScrollingBlocks(id="scrolling_block", classes="block")
            with Vertical(id="text_block", classes="block"):
                yield Digits("19.10")
                yield Digits("2025")
                yield Digits("Saturday")
            yield Label(self.LOGO_ANSI, id="logo_block", classes="block")

    def on_load(self) -> None:
        """
        Configure logging settings when the application starts.

        This method is called during the application initialization, before the
        terminal enters app mode. It sets up a custom logging format and ensures
        that all log messages are printed.

        Key actions:
            - Removes any existing log handlers
            - Adds a new log handler that prints messages with a detailed, formatted output
            - Enables capturing of log text by the main log widget

        The log format includes:
            - Timestamp (YYYY-MM-DD HH:mm:ss.SSS)
            - Log level (padded to 8 characters)
            - Module name
            - Function name
            - Line number
            - Log message
        """
        # Cause logger to print all log text. Printed text can then be  captured
        # by the main_log widget
    logger.remove()
    fmt = "{time:YYYY-MM-DD HH:mm:ss.SSS} | <level>{level: <8}</level> | {name}:{function}:{line} - {message}"
    # Only emit SUCCESS level logs to the UI (suppress INFO/DEBUG/WARNING/ERROR)
    logger.add(print, format=fmt, level="SUCCESS")

    def on_mount(self) -> None:
        """
        Mount the application and display the initial splash screen.

        This method is called when the application is first mounted, pushing the SplashScreen
        onto the screen stack to provide a welcome or loading experience for the user before
        transitioning to the main application interface.

        Returns:
            None: Does not return any value, simply initializes the splash screen.
        """
        # Display the splash screen for a few moments
        self.push_screen(SplashScreen())

    def action_help(self) -> None:
        """Someone pressed the help key!."""
        self.push_screen(HelpScreen(id="help_screen"))

    def on_key(self, event: events.Key) -> None:
        """ "A key is pressed."""
        logger.debug(f"Pressed {event.character}")
        logger.info("some warning")

    def action_quit(self) -> None:  # type: ignore
        """
        Quit the application and exit with a status code of 0.

        This method terminates the current Textual application instance,
        effectively closing the terminal user interface.

        Note:
            - The commented-out `self.glados.cancel()` suggests a potential future implementation
                for cancelling background tasks before exiting.
            - Uses `exit(0)` to indicate a successful, intentional application termination.

        Raises:
            SystemExit: Exits the application with a zero status code.
        """
        # self.glados.cancel()
        self.exit()

    def start_glados(self) -> None:
        """
        Start the GLaDOS worker thread in the background.

        This method initializes a worker thread to run the GLaDOS module's start function.
        The worker is run exclusively and in a separate thread to prevent blocking the main application.

        Notes:
            - Uses `run_worker` to create a non-blocking background task
            - Sets the worker as an instance attribute for potential later reference
            - The `exclusive=True` parameter ensures only one instance of this worker runs at a time
        """

        config_path = "configs/glados_config.yaml"
        glados_config = GladosConfig.from_yaml(str(config_path))
        glados = Glados.from_config(glados_config)

        # Keep a reference to the Glados instance so we can call control methods like start_manual_listening
        self._glados_instance = glados

        # Run the listening loop in a background worker
        self.glados_worker = self.run_worker(glados.start_listen_event_loop, exclusive=False, thread=True)

    @classmethod
    def run_app(cls, config_path: str | Path = "glados_config.yaml") -> None:
        """Class method to create and run the app instance."""
        try:
            app = cls()
            app.run()
        except KeyboardInterrupt:
            sys.exit()

    def action_start_listening(self) -> None:
        """User-initiated manual trigger to start listening (spacebar)."""
        # If we have a Glados instance, call its manual trigger method. Otherwise, log.
        if hasattr(self, "_glados_instance") and self._glados_instance is not None:
            try:
                self._glados_instance.start_manual_listening()
            except Exception:
                logger.exception("Failed to trigger manual listening on Glados instance")
        else:
            logger.warning("Glados instance not running; cannot start manual listening")


if __name__ == "__main__":
    GladosUI.run_app()
