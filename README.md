# AI Tour Guide

An intelligent, location-aware tour guide system that uses GPS data, Points of Interest (POI) queries, and AI-powered language models to provide real-time audio tour information as you travel.

## Features

- **Real-time GPS Tracking**: Supports both gpsd daemon and direct NMEA serial GPS devices
- **Intelligent POI Detection**: Automatically detects nearby points of interest using efficient spatial indexing
- **AI-Powered Narratives**: Uses MLC-LLM to generate contextual, informative tour guide responses
- **Text-to-Speech**: Hybrid TTS system supporting multiple platforms (macOS, Linux/Raspberry Pi)
- **Smart Cooldown System**: Prevents repetitive alerts for the same POI
- **Cross-Platform**: Automatically detects and adapts to Raspberry Pi or development environments

## Architecture

The system consists of several key components:

```
┌─────────────┐
│   GPS Data  │ (gpsd or NMEA serial)
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ GPS Parser  │ (core/gpsd_parser.py)
└──────┬──────┘
       │
       ▼
┌─────────────┐     ┌──────────────┐
│ POI Query   │────▶│ Find Nearest │
│   Engine    │     │     POI      │
└──────┬──────┘     └──────────────┘
       │
       ▼
┌─────────────┐
│   LLM       │ (MLC-LLM inference)
│  Inference  │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│     TTS     │ (pyttsx3/espeak)
│   Engine    │
└─────────────┘
```

## Prerequisites

### System Requirements

- **Python**: 3.9 or higher (3.11 recommended)
- **GPS Hardware**: GPS receiver with serial output or gpsd daemon
- **Audio Output**: Speakers or audio device for TTS output
- **Operating System**: Linux (Raspberry Pi recommended) or macOS for development

### System Dependencies

**Linux (Raspberry Pi):**
```bash
sudo apt-get update
sudo apt-get install -y gpsd gpsd-clients espeak espeak-data ffmpeg alsa-utils
```

**macOS:**
- Built-in `say` command is used (no additional installation needed)
- For development, ensure Xcode Command Line Tools are installed

## Installation

### Local Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd AI-TOUR-GUIDE
   ```

2. **Create a virtual environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   
   **Note**: For MLC-LLM, install from official wheels:
   ```bash
   pip install --pre -U -f https://mlc.ai/wheels mlc-llm-nightly-cpu mlc-ai-nightly-cpu
   ```

4. **Set up GPS (if using hardware)**
   
   For Raspberry Pi with gpsd:
   ```bash
   sudo systemctl enable gpsd
   sudo systemctl start gpsd
   ```
   
   For direct NMEA serial:
   - Ensure your GPS device is connected and accessible (e.g., `/dev/ttyAMA0` or `/dev/ttyUSB0`)
   - Update `GPS_SERIAL_PORT` in `core/main.py` if needed

5. **Configure POI data**
   - Edit `data/poi.json` to add your points of interest
   - Format: `"[-84.3946, 33.7634]": {"name": "Location Name", "city": "City Name"}`

### Docker Installation

The Docker image includes a full MLC-LLM setup with all required system dependencies. MLC-LLM is automatically installed from official mlc.ai wheels during the build process.

1. **Build the Docker image**
   ```bash
   docker compose build
   ```
   
   **Note**: The first build may take several minutes as it downloads and installs MLC-LLM dependencies.

2. **Configure device access** (if using GPS hardware)
   
   Edit `compose.yaml` and uncomment/modify the `devices:` section:
   ```yaml
   devices:
     - /dev/ttyAMA0:/dev/ttyAMA0  # Raspberry Pi GPS
   ```

3. **Set environment variables** (optional)
   
   Create a `.env` file or modify `compose.yaml`:
   ```bash
   GPS_SERIAL_PORT=/dev/ttyAMA0
   ALERT_DISTANCE_MILES=0.25
   POI_COOLDOWN_MINUTES=30
   ```

4. **Run the container**
   ```bash
   docker compose up
   ```

   Or run in detached mode:
   ```bash
   docker compose up -d
   ```

5. **View logs**
   ```bash
   docker compose logs -f tour-guide
   ```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GPS_SERIAL_PORT` | `/dev/ttyAMA0` (Pi) or `/dev/ttys009` (macOS) | Serial port for GPS device |
| `ALERT_DISTANCE_MILES` | `0.25` | Distance threshold for POI alerts (miles) |
| `POI_COOLDOWN_MINUTES` | `30` | Minutes before re-alerting the same POI |
| `GPS_TIMEOUT_SECONDS` | `10` | Seconds before warning about GPS timeout |

### POI Data Format

Edit `data/poi.json` to customize points of interest:

```json
{
    "[-84.3946, 33.7634]": {
        "name": "Georgia Aquarium",
        "city": "Atlanta"
    },
    "[-84.3928, 33.7625]": {
        "name": "World of Coca-Cola",
        "city": "Atlanta"
    }
}
```

Coordinates are in `[longitude, latitude]` format.

### LLM Model Configuration

The system uses MLC-LLM models. By default, it loads from HuggingFace:
- Model: `HF://mlc-ai/SmolLM2-135M-Instruct-q0f32-MLC`

To use a local model, edit `llm/llm_inference.py` and change `MODEL_PATH`.

## Usage

### Running Locally

```bash
python3 core/main.py
```

The system will:
1. Initialize GPS connection
2. Load POI database
3. Start listening for GPS updates
4. Automatically announce nearby POIs with AI-generated descriptions

### Running with Docker

```bash
docker compose up
```

### Stopping the Application

Press `Ctrl+C` to gracefully stop the application.

## Development

### Project Structure

```
AI-TOUR-GUIDE/
├── audio/              # Text-to-speech module
│   └── tts.py         # HybridTTS implementation
├── core/              # Core application logic
│   ├── main.py        # Main entry point
│   ├── gpsd_parser.py # GPS data parsing
│   ├── poi_query.py   # POI spatial queries
│   └── virtual_gps.py # Virtual GPS for testing
├── data/              # Data files
│   └── poi.json       # Points of Interest database
├── llm/               # LLM inference
│   ├── llm_inference.py
│   ├── prompt.txt     # System prompt template
│   └── model/         # LLM model files
├── benchmarking/      # Performance benchmarking tools
├── scripts/           # Build and run scripts
├── Dockerfile         # Docker container definition
├── compose.yaml       # Docker Compose configuration
└── requirements.txt   # Python dependencies
```

### Testing Components

**Test LLM inference:**
```bash
python3 llm/llm_inference.py
```

**Test TTS:**
```bash
python3 audio/tts.py
```

**Test GPS parser:**
```bash
python3 core/gpsd_parser.py
```

**Test virtual GPS (for development without hardware):**
```bash
python3 core/virtual_gps.py
```

### Benchmarking

The project includes benchmarking tools for performance analysis:

```bash
cd benchmarking
./run_benchmark.sh
```

Results are saved in `benchmarking/results/` and analysis in `benchmarking/analysis/`.

## Troubleshooting

### GPS Not Working

**Issue**: No GPS data received

**Solutions**:
- Check GPS device connection and permissions
- Verify gpsd is running: `sudo systemctl status gpsd`
- Test GPS manually: `gpsmon` or `cgps`
- Check device path in configuration
- Ensure user has access to serial device (may need to add user to `dialout` group)

### TTS Not Working

**Issue**: No audio output

**Solutions**:
- **Linux**: Install `espeak` or `alsa-utils`
- **macOS**: Check system audio settings
- Verify audio device is connected and unmuted
- Check logs for TTS errors

### LLM Model Not Loading

**Issue**: Model initialization fails

**Solutions**:
- Check internet connection (if loading from HuggingFace)
- Verify model path in `llm/llm_inference.py`
- Ensure sufficient disk space for model files
- **Local installation**: Verify MLC-LLM installation: `python -c "import mlc_llm; print(mlc_llm.__path__)"`
- **Docker**: Check build logs for MLC-LLM installation errors: `docker compose build --no-cache`
- **Docker**: Verify MLC-LLM in container: `docker compose run tour-guide python -c "import mlc_llm"`

### POI Not Detected

**Issue**: Nearby POIs not triggering alerts

**Solutions**:
- Verify POI coordinates in `data/poi.json` are correct
- Check `ALERT_DISTANCE_MILES` setting (may be too small)
- Verify GPS coordinates are valid (check logs)
- Ensure POI file is readable

### Docker Issues

**Issue**: Container won't start or GPS not accessible

**Solutions**:
- Ensure device paths are correct in `compose.yaml`
- Try using `privileged: true` mode (less secure)
- Check Docker has access to devices: `docker info`
- Verify volumes are mounted correctly
- Check container logs: `docker compose logs tour-guide`
- **MLC-LLM build issues**: If build fails, try rebuilding without cache: `docker compose build --no-cache`
- **MLC-LLM runtime errors**: Check that all system dependencies installed correctly in build logs

## Performance Considerations

- **LLM Inference**: First inference may be slow (model loading). Subsequent calls are faster.
- **POI Queries**: Uses Numba-accelerated Haversine distance calculations for real-time performance.
- **TTS**: Audio generation is queued to prevent blocking GPS processing.
- **Memory**: Model requires ~500MB-1GB RAM depending on quantization.

## License

See [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please ensure:
- Code follows existing style
- Tests pass (if applicable)
- Documentation is updated

## Credits

- **MLC-LLM**: Model inference engine
- **gpsd**: GPS daemon for hardware abstraction
- **pyttsx3**: Cross-platform TTS library
- **Numba**: High-performance numerical computing

## Support

For issues, questions, or contributions, please open an issue on the repository.
