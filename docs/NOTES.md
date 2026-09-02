# NOTES.md

## Day 2 — 2026-08-31

**Had to install nvidia-jetpack first.** Error message occured, Python couldn't find cv2 at
all. Turns out flashing the Jetson only gives you Ubuntu — OpenCV and
CUDA come separately with `sudo apt install nvidia-jetpack`. ~40 min
download. There are 83 other updates waiting, ignoring them for now
as they're not crucial.

**Camera:** Logitech 720p, plugged into USB. Shows up as /dev/video0.
There's also a video1 but it gives no frames, it's just metadata.

Attempted the res 1920x1080 but it just gave me 1280x720 @ 30fps silently

**Format comparison (1280x720):**

| Format | Requested fps | Actual fps |
|------|----|----|
| MJPG | 30 | 30 |
| YUYV | 30 | 10 |

YUYV caps at 10fps because uncompressed 720p needs ~55MB/s and USB 2.0
only carries ~35-40. MJPG compresses on the camera so less data crosses
the wire — but my Jetson has to decode every frame. Going with MJPG.

**How the capture loop is set up:** open the camera (had to state
CAP_V4L2 explicitly or OpenCV picks an unusual format), I had to set the
format before the resolution and loop to read frames. Always remember to
call `cap.release()` at the end; otherwise, the camera remains locked,
and a subsequent run will fail. I also added a `--headless` flag to save
frames to /tmp since the display window doesn’t work over SSH.

**The corrupt JPEG spam is fine.** It says "extraneous bytes before
marker 0xd9" on almost every frame. The camera just adds junk to the
end of each JPEG and the decoder complains. Overall not worth fixing.
Would have to consider where I will send all this output somewhere else though

**IntelliJ setup issues:** all my files live on the Jetson, my laptop
is basically just a screen. GitHub Desktop can't see any of it, so all
git happens in the IntelliJ terminal. Lost my connection once and it doesn't
save anything until I reconnect. Noted to never close the connection between the
two.


**Git:** my commits weren't showing up on my contribution graph.
GitHub matches commits by email, not name, and mine was wrong. Fixed
with `git config --global user.email`. Also passwords don't work for
pushing anymore, need a personal access token.

**Still don't know:**
- Is the MJPEG decoding what's slowing things down?
- Does the camera come back as video0 after unplugging it?

## Day 3 - 9/1/2026

Spent the day trying to get a real AI model (SSD-MobileNet) running through
jetson-inference and never got it to build. Instead four different errors in a
row when attempting to download, all from the same thing: my Jetson shipped with much
newer versions of CUDA and Python than that project was written for. Fixed each one and
another appeared behind it, so I stopped after about two hours.

Decided to use OpenCV's built-in HOG people detector since it needs no
install. It ran but only found me about 1 out of every 5 tries no matter
where I stood, and took ~400ms per frame. Loosening the parameters but it increased the
detections to about 10, most of which were false positives.

Tried motion detection (MOG2) instead. Way faster at ~3.5ms, but it split
one person into about 5 separate boxes and the boxes were way off from
my actual position in the frame.

Ended up scrapping all of it. I went straight to a real model using cv2.dnn instead. Download
three files (config, weights, class names) was able to load them without any issues.
Used the older Darknet format on purpose since OpenCV 4.6's ONNX reader can reject newer YOLO exports.

**Comparison:**

| | HOG | YOLOv4-tiny |
|------|----|----|
| Time per frame | ~400ms | ~100ms |
| Found me | 1 in 5 | 1 in 1 |
| False positives | many | occasionally |
| Classes | person only | 80 |


**Lesson:** I lost hours on a build that was never going to work with my
JetPack version. The cv2.dnn route needed zero setup and I could have started
there. Next time something needs compiling against specific CUDA or Python
versions, check first whether there's a route that just loads a file instead.

**Still don't know:**
- Does the model hold up in worse lighting?
- How much does the frame quality matter?
