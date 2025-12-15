sudo apt-get install ccache ninja-build
sudo sysctl kernel.perf_event_paranoid=1
pip install uv
uv venv
source venv/bin/activate
pip install matplotlib seaborn pandas numpy