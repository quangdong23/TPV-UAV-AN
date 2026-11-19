"""Start RTMP analysis; add --publish to send AI video to configured RTMP output."""
import sys
import os
import time
os.environ['TZ'] = 'Asia/Ho_Chi_Minh'
if hasattr(time, 'tzset'):
    time.tzset()
from pathlib import Path
from uav_pipeline.cli.run import main

if __name__ == '__main__':
    root=Path(__file__).resolve().parent
    url=(root/'configs/rtmp_input.txt').read_text().strip()
    arguments=list(sys.argv[1:])
    publish=[]
    if '--publish' in arguments:
        arguments.remove('--publish')
        destination=(root/'configs/rtmp_output.txt').read_text().strip()
        publish=['--live-output',destination]
        print('[live-output] Annotated publishing enabled; endpoint must be available on the video server.',flush=True)
    raise SystemExit(main(['--input',url,'--events','--upload-events',*publish,*arguments]))
