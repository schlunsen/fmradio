import asyncio
import numpy as np
import sounddevice as sd
import scipy.signal as signal
from rtlsdr import RtlSdr
from typing import Optional, List, Callable

from .stations import StationManager, RadioStation

# SDR and DSP parameters
DEFAULT_SAMPLE_RATE = 2.048e6  # Sample rate (e.g., 2.048 Msps). Lower rates can reduce CPU.
SDR_CENTER_FREQ_OFFSET = 0  # No offset by default, tune directly. Can be 250e3 to avoid DC spike.
FM_BANDWIDTH = 200e3  # Bandwidth of an FM signal (approx 200 kHz)
TARGET_AUDIO_SAMPLE_RATE = 48000  # Desired audio output sample rate
SDR_BLOCK_SIZE = 1024 * 32  # SDR Sample block size

# Scanning parameters
SCAN_STEP = 0.1  # MHz
SCAN_DWELL_TIME = 0.5  # seconds
SIGNAL_THRESHOLD = 0.1  # Signal strength threshold for station detection

class RadioController:
    """
    Handles SDR operations, signal processing, and audio playback.
    Runs its core processing in separate threads to avoid blocking the TUI.
    """
    def __init__(self):
        self.sdr = None
        self.audio_stream = None
        self.is_playing = False
        self.is_scanning = False
        self.target_freq_hz = 0
        self.actual_sdr_center_freq_hz = 0
        self.on_status_update = None  # Callback to update TUI status
        self.on_station_found = None  # Callback for when a station is found during scanning
        self._stop_requested = asyncio.Event()
        self.app_instance = None  # To call app.call_from_thread
        self.station_manager = StationManager()

    def set_status_updater(self, updater_callback):
        """Sets the callback function for updating the TUI status."""
        self.on_status_update = updater_callback

    def set_station_found_callback(self, callback: Callable[[RadioStation], None]):
        """Sets the callback function for when a station is found during scanning."""
        self.on_station_found = callback

    def _update_status(self, message: str):
        """Helper to call the status update callback if it's set."""
        if self.on_status_update:
            self.on_status_update(message)

    def _signal_strength(self, samples_iq: np.ndarray) -> float:
        """Calculate the signal strength of the current frequency."""
        if len(samples_iq) == 0:
            return 0.0
        
        # Calculate power spectrum
        spectrum = np.abs(np.fft.fft(samples_iq))
        # Get the center frequency power
        center_idx = len(spectrum) // 2
        bandwidth_idx = int(FM_BANDWIDTH / (self.sdr.sample_rate / len(spectrum)))
        
        # Calculate average power in the FM bandwidth
        start_idx = center_idx - bandwidth_idx // 2
        end_idx = center_idx + bandwidth_idx // 2
        signal_power = np.mean(np.abs(spectrum[start_idx:end_idx]))
        
        return signal_power

    async def scan_frequency_range(self, start_freq: float, end_freq: float) -> List[RadioStation]:
        """Scan a frequency range for stations."""
        if not self.sdr:
            self._update_status("SDR not initialized. Cannot scan.")
            return []

        if self.is_playing:
            await self.stop_streaming_async()

        self.is_scanning = True
        self._stop_requested.clear()
        found_stations = []

        try:
            current_freq = start_freq
            while current_freq <= end_freq and not self._stop_requested.is_set():
                self._update_status(f"Scanning {current_freq:.1f} MHz...")
                
                # Tune to frequency
                self.sdr.center_freq = current_freq * 1e6
                
                # Read samples and check signal strength
                samples = self.sdr.read_samples(SDR_BLOCK_SIZE)
                signal_strength = self._signal_strength(samples)
                
                if signal_strength > SIGNAL_THRESHOLD:
                    # Check if we already know this station
                    known_station = self.station_manager.get_station_by_frequency(current_freq)
                    if known_station:
                        found_stations.append(known_station)
                        if self.on_station_found:
                            self.on_station_found(known_station)
                    else:
                        # Create a new station entry
                        new_station = RadioStation(f"Unknown Station", current_freq)
                        found_stations.append(new_station)
                        if self.on_station_found:
                            self.on_station_found(new_station)
                
                await asyncio.sleep(SCAN_DWELL_TIME)
                current_freq += SCAN_STEP

        except Exception as e:
            self._update_status(f"Error during scanning: {e}")
        finally:
            self.is_scanning = False
            self._update_status("Scanning complete.")

        return found_stations

    def get_known_stations(self) -> List[RadioStation]:
        """Get the list of known stations."""
        return self.station_manager.get_all_stations()

    def init_sdr(self) -> bool:
        """Initializes the RTL-SDR dongle."""
        if self.sdr:
            self.close_sdr() 
        try:
            self._update_status("Attempting to initialize SDR...")
            self.sdr = RtlSdr()
            self.sdr.sample_rate = DEFAULT_SAMPLE_RATE
            self.sdr.gain = 'auto'
            _ = self.sdr.read_samples(256) 
            self._update_status(f"SDR Initialized. Sample Rate: {self.sdr.sample_rate/1e6:.3f} Msps.")
            return True
        except Exception as e:
            self.sdr = None 
            self._update_status(f"Error initializing SDR: {e}. Is it plugged in? Is librtlsdr installed?")
            return False

    def tune_to_frequency(self, freq_mhz: float) -> bool:
        """Sets the target frequency for the SDR."""
        if not self.sdr:
            self._update_status("SDR not initialized. Cannot tune.")
            return False

        self.target_freq_hz = freq_mhz * 1e6
        self.actual_sdr_center_freq_hz = self.target_freq_hz - SDR_CENTER_FREQ_OFFSET

        try:
            self.sdr.center_freq = self.actual_sdr_center_freq_hz
            self._update_status(f"Tuned SDR to center freq: {self.sdr.center_freq/1e6:.3f} MHz for target {freq_mhz:.3f} MHz.")
            return True
        except Exception as e:
            self._update_status(f"Error tuning SDR: {e}")
            return False

    def _fm_demodulate(self, samples_iq: np.ndarray) -> np.ndarray:
        """
        Performs simplified FM demodulation on complex IQ samples.
        """
        if SDR_CENTER_FREQ_OFFSET != 0:
            t = np.arange(len(samples_iq)) / self.sdr.sample_rate
            mixer = np.exp(-1j * 2 * np.pi * (-SDR_CENTER_FREQ_OFFSET) * t)
            samples_iq = samples_iq * mixer

        num_taps_channel = 65 
        cutoff_channel = FM_BANDWIDTH / 2 
        
        if cutoff_channel >= (self.sdr.sample_rate / 2):
            self._update_status(f"Warning: Channel cutoff ({cutoff_channel/1e3}kHz) too high. Skipping channel filter.")
            filtered_iq = samples_iq 
        else:
            fir_coeffs_channel = signal.firwin(num_taps_channel, cutoff_channel, fs=self.sdr.sample_rate, window='hamming')
            filtered_iq = signal.lfilter(fir_coeffs_channel, 1.0, samples_iq)
        
        decimation_factor_pre_demod = int(self.sdr.sample_rate / (FM_BANDWIDTH * 1.5))
        if decimation_factor_pre_demod < 1: decimation_factor_pre_demod = 1

        if decimation_factor_pre_demod > 1:
            demod_input_iq = signal.decimate(filtered_iq, decimation_factor_pre_demod, ftype='fir', zero_phase=True)
            current_sample_rate = self.sdr.sample_rate / decimation_factor_pre_demod
        else:
            demod_input_iq = filtered_iq 
            current_sample_rate = self.sdr.sample_rate

        if len(demod_input_iq) < 2: 
            return np.array([], dtype=np.float32) 
        
        conj_product = demod_input_iq[1:] * np.conj(demod_input_iq[:-1])
        demodulated_signal = np.angle(conj_product)

        time_constant_deemphasis = 75e-6 
        alpha_deemphasis = np.exp(-1.0 / (current_sample_rate * time_constant_deemphasis))
        demodulated_signal = signal.lfilter([1.0 - alpha_deemphasis], [1.0, -alpha_deemphasis], demodulated_signal)

        num_audio_samples = int(len(demodulated_signal) * TARGET_AUDIO_SAMPLE_RATE / current_sample_rate)
        
        if num_audio_samples > 0 and len(demodulated_signal) > 1 : 
            try:
                audio_output = signal.resample(demodulated_signal, num_audio_samples)
            except ValueError: 
                 audio_output = np.array([], dtype=np.float32) 
        else:
            audio_output = np.array([], dtype=np.float32) 
        
        if audio_output.size > 0:
            max_abs = np.max(np.abs(audio_output))
            if max_abs > 1e-9: 
                audio_output = (audio_output / max_abs) * 0.5 
            else:
                audio_output = np.zeros_like(audio_output) 
        
        return audio_output.astype(np.float32) 

    def _sdr_read_and_process_loop(self):
        """
        Sets up the asynchronous sample reading from the SDR.
        Processing happens in `sdr_callback`.
        """
        def sdr_callback(samples_iq_chunk: np.ndarray, context: RadioController):
            if context._stop_requested.is_set() or not context.is_playing:
                return

            try:
                audio_data = context._fm_demodulate(samples_iq_chunk)
                if audio_data.size > 0 and context.audio_stream and context.is_playing:
                    context.audio_stream.write(audio_data) 
            except Exception as e:
                context._update_status(f"DSP Error: {e}")

        self.sdr.read_samples_async(sdr_callback, num_samples=SDR_BLOCK_SIZE, context=self)

    async def start_streaming_async(self) -> bool:
        """Starts SDR streaming and audio playback asynchronously."""
        if not self.sdr or self.target_freq_hz == 0:
            self._update_status("SDR not ready or frequency not set.")
            return False
        if self.is_playing:
            await self.stop_streaming_async()

        self._update_status(f"Starting stream for {self.target_freq_hz/1e6:.3f} MHz...")
        self._stop_requested.clear() 
        self.is_playing = True

        try:
            if abs(self.sdr.center_freq - self.actual_sdr_center_freq_hz) > 1: 
                self.sdr.center_freq = self.actual_sdr_center_freq_hz

            self.audio_stream = sd.OutputStream(
                samplerate=TARGET_AUDIO_SAMPLE_RATE,
                channels=1, 
                dtype='float32', 
                blocksize=0, 
                latency='low' 
            )
            self.audio_stream.start() 
            
            self._sdr_read_and_process_loop() 
            
            self._update_status(f"Streaming live from {self.target_freq_hz/1e6:.3f} MHz.")
            return True
        except Exception as e:
            self._update_status(f"Error starting stream: {e}")
            self.is_playing = False 
            if self.audio_stream:
                try:
                    self.audio_stream.stop()
                    self.audio_stream.close()
                except Exception as audio_err:
                    self._update_status(f"Error cleaning up audio stream: {audio_err}")
                self.audio_stream = None
            return False

    async def stop_streaming_async(self):
        """Stops SDR streaming and audio playback asynchronously."""
        if not self.is_playing and \
           (not self.sdr or \
            not hasattr(self.sdr, 'async_handler_thread') or \
            (hasattr(self.sdr, 'async_handler_thread') and self.sdr.async_handler_thread is None)):
            self._update_status("Not currently streaming or SDR not in async mode.")
            return

        self._update_status("Stopping stream...")
        self._stop_requested.set() 

        if self.sdr and hasattr(self.sdr, 'cancel_read_async'):
            try:
                if hasattr(self.sdr, 'async_handler_thread') and \
                   self.sdr.async_handler_thread is not None and \
                   self.sdr.async_handler_thread.is_alive():
                    self.sdr.cancel_read_async()
                    await asyncio.sleep(0.3) 
            except Exception as e:
                self._update_status(f"Note: Error during cancel_read_async (may be benign): {e}")
        
        if self.audio_stream:
            try:
                self.audio_stream.stop()  
                self.audio_stream.close() 
            except Exception as e:
                self._update_status(f"Error stopping audio stream: {e}")
            self.audio_stream = None 
        
        self.is_playing = False 
        self._update_status("Streaming stopped.")

    def close_sdr(self):
        """Closes the SDR device and cleans up resources."""
        if self.is_playing:
             self._update_status("Warning: close_sdr called while still in playing state.")

        if self.sdr:
            try:
                self._update_status("Closing SDR device...")
                sdr_to_close = self.sdr
                self.sdr = None 
                sdr_to_close.close() 
                self._update_status("SDR closed.")
            except Exception as e:
                self._update_status(f"Error closing SDR: {e}")
        self.sdr = None 