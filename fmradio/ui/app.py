from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Header, Footer, Static, Input, Button, DataTable
from textual.reactive import reactive
from textual.events import Key
from pathlib import Path

from ..core.radio_controller import RadioController
from ..core.stations import RadioStation

class RadioApp(App):
    """Textual TUI Application for the FM Radio Listener."""

    CSS_PATH = "radio_app_styles.css" 
    TITLE = "Command-Line FM Radio (RTL-SDR)"

    BINDINGS = [
        ("q", "quit_app", "Quit Radio"),
        ("s", "start_scan", "Start Scan"),
        ("c", "stop_scan", "Stop Scan")
    ]

    status_line = reactive("Welcome! Initialize SDR and tune to an FM station.")
    found_stations = reactive([])

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.radio_ctrl = RadioController()
        self.radio_ctrl.app_instance = self
        self.radio_ctrl.set_status_updater(self.update_status_from_controller)
        self.radio_ctrl.set_station_found_callback(self.on_station_found)

    def compose(self) -> ComposeResult:
        """Create child widgets for the app's layout."""
        yield Header() 
        with Vertical(id="main_container"): 
            yield Static(id="status_display")
            with Horizontal(id="input_area"): 
                yield Input(placeholder="Frequency (MHz, e.g., 100.7)", id="freq_input", type="number")
                yield Button("Tune / Play", id="tune_play_button", variant="primary")
            yield Button("Stop Stream", id="stop_button", variant="error")
            with Horizontal(id="scan_area"):
                yield Button("Start Scan", id="scan_button", variant="primary")
                yield Button("Stop Scan", id="stop_scan_button", variant="error")
            yield DataTable(id="stations_table", zebra_stripes=True)
        yield Footer() 

    def on_mount(self) -> None:
        """Called when the app is first mounted."""
        self.query_one("#status_display", Static).update(self.status_line)
        self.run_worker(self.initialize_sdr_task(), thread=True, group="sdr_ops", exclusive=True)
        
        # Initialize the stations table
        table = self.query_one("#stations_table", DataTable)
        table.add_columns("Station", "Frequency", "Description")
        self._update_station_list()

    def _update_station_list(self):
        """Update the stations table."""
        stations = self.radio_ctrl.get_known_stations()
        table = self.query_one("#stations_table", DataTable)
        table.clear()
        
        for station in stations:
            table.add_row(
                station.name,
                f"{station.frequency:.1f} MHz",
                station.description or ""
            )

    def on_station_found(self, station: RadioStation):
        """Called when a station is found during scanning."""
        self.call_from_thread(self._add_found_station, station)

    def _add_found_station(self, station: RadioStation):
        """Add a found station to the list."""
        if station not in self.found_stations:
            self.found_stations.append(station)
            self._update_station_list()

    def watch_found_stations(self, stations):
        """Called when found_stations changes."""
        self._update_station_list()

    async def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        """Handle station selection from the table."""
        row = event.row
        if row is not None:
            # Extract frequency from the row (it's in the second column)
            freq_str = row[1].replace(" MHz", "")
            try:
                freq_val = float(freq_str)
                freq_input = self.query_one("#freq_input", Input)
                freq_input.value = str(freq_val)
                await self.on_button_pressed(Button.Pressed(self.query_one("#tune_play_button", Button)))
            except ValueError:
                self.status_line = "Error parsing frequency from selected station."

    async def on_key(self, event: Key) -> None:
        """Handle key events."""
        if event.key == "enter":
            # Check if the DataTable has focus
            table = self.query_one("#stations_table", DataTable)
            if table.has_focus:
                cursor_row = table.cursor_row
                if cursor_row is not None:
                    # Get the frequency from the selected row
                    freq_str = table.get_row_at(cursor_row)[1].replace(" MHz", "")
                    try:
                        freq_val = float(freq_str)
                        freq_input = self.query_one("#freq_input", Input)
                        freq_input.value = str(freq_val)
                        await self.on_button_pressed(Button.Pressed(self.query_one("#tune_play_button", Button)))
                    except ValueError:
                        self.status_line = "Error parsing frequency from selected station."

    async def action_start_scan(self) -> None:
        """Start scanning for stations."""
        self.run_worker(self.scan_task(), thread=True, group="sdr_ops", exclusive=True)

    async def action_stop_scan(self) -> None:
        """Stop scanning for stations."""
        self.radio_ctrl._stop_requested.set()

    async def scan_task(self) -> None:
        """Task to handle scanning."""
        if not self.radio_ctrl.sdr:
            self.status_line = "SDR not initialized. Cannot scan."
            return

        self.status_line = "Starting scan..."
        found_stations = await self.radio_ctrl.scan_frequency_range(87.5, 108.0)
        if found_stations:
            self.status_line = f"Found {len(found_stations)} stations."
        else:
            self.status_line = "No stations found."

    def update_status_from_controller(self, message: str):
        """Thread-safe way for RadioController to update the status_line."""
        self.call_from_thread(self._set_status_line, message)

    def _set_status_line(self, message: str):
        """Internal method to set status_line, ensuring it runs in the main thread."""
        self.status_line = message

    def watch_status_line(self, new_status: str) -> None:
        status_widget = self.query_one("#status_display", Static)
        status_widget.update(new_status)

    async def initialize_sdr_task(self) -> None:
        """Task to initialize SDR. Runs in a worker thread."""
        self.status_line = "Initializing SDR, please wait..."
        if not self.radio_ctrl.init_sdr():
            self.status_line = "SDR Initialization Failed. Check console. Dongle connected? Drivers (librtlsdr) installed?"
        else:
            self.status_line = f"SDR Ready (Rate: {self.radio_ctrl.sdr.sample_rate/1e6:.2f} Msps). Enter FM Freq and click Tune."

    async def tune_and_play_task(self, freq_mhz: float) -> None:
        """Task to handle tuning and starting playback. Runs in a worker thread."""
        if not self.radio_ctrl.sdr:
            self.status_line = "SDR not initialized. Cannot tune."
            return

        self.status_line = f"Tuning to {freq_mhz:.2f} MHz..."
        if self.radio_ctrl.tune_to_frequency(freq_mhz):
            self.status_line = f"Attempting to stream {freq_mhz:.2f} MHz..."
            if not await self.radio_ctrl.start_streaming_async():
                self.status_line = f"Failed to start stream for {freq_mhz:.2f} MHz. Check console."
        else:
            self.status_line = f"Failed to tune to {freq_mhz:.2f} MHz."

    async def stop_stream_task(self) -> None:
        """Task to stop the stream. Runs in a worker thread."""
        self.status_line = "Processing stop request..."
        await self.radio_ctrl.stop_streaming_async()

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button press events."""
        if event.button.id == "tune_play_button":
            freq_input_widget = self.query_one("#freq_input", Input)
            try:
                freq_val = float(freq_input_widget.value)
                if not (87.0 <= freq_val <= 108.0 or 65.0 <= freq_val <= 74.0 or 76.0 <= freq_val <= 95.0): 
                    self.status_line = "Frequency out of typical FM bands (e.g., 87.5-108.0 MHz)."
                self.run_worker(self.tune_and_play_task(freq_val), thread=True, group="sdr_ops", exclusive=True)
            except ValueError:
                self.status_line = "Invalid frequency format. Please use numbers (e.g., 100.7)."
        elif event.button.id == "stop_button":
            self.run_worker(self.stop_stream_task(), thread=True, group="sdr_ops", exclusive=True)
        elif event.button.id == "scan_button":
            await self.action_start_scan()
        elif event.button.id == "stop_scan_button":
            await self.action_stop_scan()

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle Enter key in the input field."""
        if event.input.id == "freq_input":
            tune_button = self.query_one("#tune_play_button", Button)
            await self.on_button_pressed(Button.Pressed(tune_button))

    async def shutdown_task(self):
        """Task to handle shutdown procedures. Runs in a worker thread."""
        self.status_line = "Shutting down radio controller..."
        await self.radio_ctrl.stop_streaming_async() 
        self.radio_ctrl.close_sdr() 
        self.status_line = "Shutdown complete. Exiting."
        self.call_from_thread(self.exit, "Radio resources released.")

    async def action_quit_app(self) -> None:
        """Called when 'q' is pressed or quit is triggered."""
        self.status_line = "Initiating shutdown..."
        self.run_worker(self.shutdown_task(), thread=True, group="sdr_shutdown", exclusive=True) 