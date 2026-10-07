Stochastic Nonlinearities Improve Uncertainty Estimation — BMVC 2026 project page

FILES
- index.html
- style.css
- script.js
- paper.pdf
- assets/

YOUTUBE
Open script.js and replace:
  const YOUTUBE_VIDEO_ID = 'YOUR_YOUTUBE_VIDEO_ID';
with the ID from your YouTube URL.
Example:
  https://www.youtube.com/watch?v=abc123XYZ
becomes:
  const YOUTUBE_VIDEO_ID = 'abc123XYZ';

OFFICIAL LOGOS
The Bilkent University, NeuraVision Lab, and BMVC 2026 marks in assets/ were taken from the supplied BMVC presentation assets. They are not recreated wordmarks.

RESULT TABLES
The quantitative-results carousel contains the main paper tables:
1. Segmentation performance
2. Segmentation uncertainty & calibration
3. Cross-dataset segmentation OOD
4. Classification accuracy & OOD detection

The old inference-pass slider/section has been removed.


VIDEO PREVIEW
-------------
The YouTube player is embedded from:
https://www.youtube.com/watch?v=jEi-fqd9yHQ

If you open index.html directly with file://, YouTube may refuse the iframe with
player error 153 because no HTTP referrer is available. The page now shows a clean
YouTube fallback in that case. To preview the real embedded player locally, serve
the folder over HTTP, for example:

    python -m http.server 8000

Then open http://localhost:8000/ in your browser. On a deployed HTTPS website, the
embedded player is used automatically. If YouTube still reports that playback is
unavailable after deployment, verify that "Allow embedding" is enabled for the video.
