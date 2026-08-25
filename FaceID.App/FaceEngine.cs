// FaceEngine.cs
//
// Thin wrapper around OpenCV's YuNet detector and SFace recognizer.
//
//   FaceDetectorYN   -> finds faces in an image (returns bounding boxes + 5 landmarks)
//   FaceRecognizerSF -> turns one aligned face into a fixed-length embedding vector,
//                       plus a built-in cosine similarity helper
//
using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Dnn;
using Emgu.CV.Face;
using Emgu.CV.Structure;
using Emgu.CV.Util;

namespace FaceTool;

/// <summary>Everything we know about one face found in one image.</summary>
public sealed record FaceBox(
    int X, int Y, int Width, int Height,
    float Score,
    float RightEyeX, float RightEyeY,
    float LeftEyeX, float LeftEyeY,
    float NoseX, float NoseY,
    float MouthRx, float MouthRy,
    float MouthLx, float MouthLy)
{
    public float CenterX => X + Width / 2f;
    public float CenterY => Y + Height / 2f;
}

public sealed record Subject(FaceBox Box, float Score);

public sealed record FaceRecord(
    string File,
    string Path,
    int Width,
    int Height,
    double Sharpness,          // NEW: variance of Laplacian (blur metric)
    IReadOnlyList<FaceBox> Boxes,
    Subject Subject,
    float[] Embedding);

public sealed class FaceEngine : IDisposable
{
    private readonly FaceDetectorYN _detector;
    private readonly FaceRecognizerSF _recognizer;
    private readonly Mat _image = new();
    private readonly Mat _faces = new();
    private readonly Mat _alignedFace = new();
    private readonly Mat _embedding = new();
    private bool _disposed;
    private readonly float _minFace;      // NEW: minimum face size in pixels

    // The models ship next to the binary, so resolve them against the install
    // directory rather than the caller's current working directory.
    public static readonly string DefaultDetectorModel =
        Path.Combine(AppContext.BaseDirectory, "models", "face_detection_yunet_2026may.onnx");
    public static readonly string DefaultRecognizerModel =
        Path.Combine(AppContext.BaseDirectory, "models", "face_recognition_sface_2021dec.onnx");
    private const float DetectThresholdDefault = 0.6f;

    public FaceEngine(
        string? detectorModel = null,
        string? recognizerModel = null,
        float scoreThreshold = DetectThresholdDefault,
        float minFacePixels = 48f,          // NEW: reject tiny faces by default
        Target target = Target.Cpu)   // NEW: GPU support
    {
        detectorModel ??= DefaultDetectorModel;
        recognizerModel ??= DefaultRecognizerModel;

        if (!File.Exists(detectorModel))
            throw new FileNotFoundException($"detector model not found: {detectorModel}", detectorModel);
        if (!File.Exists(recognizerModel))
            throw new FileNotFoundException($"recognizer model not found: {recognizerModel}", recognizerModel);

        _detector = new FaceDetectorYN(detectorModel, "", new Size(320, 320),
                                       scoreThreshold, 0.3f, 5000,
                                       Emgu.CV.Dnn.Backend.Default, target);
        _recognizer = new FaceRecognizerSF(recognizerModel, "", Emgu.CV.Dnn.Backend.Default, target);
        _minFace = minFacePixels;

        Console.WriteLine($"engine ready");
        Console.WriteLine($"    detector    {Path.GetFileName(detectorModel)}");
        Console.WriteLine($"    recognizer  {Path.GetFileName(recognizerModel)}");
        Console.WriteLine($"    threshold   {scoreThreshold:F2}");
        Console.WriteLine($"    min face    {minFacePixels:F0}px");
        Console.WriteLine($"    target      {target}");
    }

    /// <summary>Variance of Laplacian � low means blurry. Calibrate ~100 on your data.</summary>
    public static double Sharpness(Mat bgr)
    {
        using Mat gray = new(), lap = new();
        CvInvoke.CvtColor(bgr, gray, ColorConversion.Bgr2Gray);
        CvInvoke.Laplacian(gray, lap, DepthType.Cv64F);
        MCvScalar mean = default, stdDev = default;
        CvInvoke.MeanStdDev(lap, ref mean, ref stdDev);
        return stdDev.V0 * stdDev.V0; // sigma^2
    }

    /// <summary>Rejects faces too small to embed reliably.</summary>
    public bool IsUsable(FaceBox box) => Math.Min(box.Width, box.Height) >= _minFace;

    /// <summary>
    /// Loads an image, finds every face, picks the most central one above threshold,
    /// aligns it, embeds it. Returns null when no usable face is found.
    /// </summary>
    public FaceRecord? Analyze(string imagePath, TextWriter? log = null)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        using Mat raw = CvInvoke.Imread(imagePath, ImreadModes.ColorBgr);
        if (raw.IsEmpty)
        {
            log?.WriteLine($"        unreadable image: {imagePath}");
            return null;
        }

        int width = raw.Cols;
        int height = raw.Rows;
        double sharpness = Sharpness(raw);
        log?.WriteLine($"        image         {width}x{height}  sharpness {sharpness:F0}");

        // YuNet works on a fixed input window -> resize to cover, then scale boxes back.
        _detector.InputSize = new Size(width, height);
        raw.CopyTo(_image);
        _detector.Detect(_image, _faces);

        var boxes = ParseFaces(_faces);
        log?.WriteLine($"        faces found   {boxes.Count}");

        // NEW: quality gate � drop faces too small to be reliable. Carry each
        // face's detection-row index along, because alignment needs that row.
        var usable = boxes.Select((Box, Row) => (Box, Row))
                          .Where(f => IsUsable(f.Box))
                          .ToList();
        if (usable.Count != boxes.Count)
            log?.WriteLine($"        rejected      {boxes.Count - usable.Count} too small (<{_minFace:F0}px)");
        if (usable.Count == 0)
        {
            log?.WriteLine("        no usable face");
            return null;
        }
        boxes = usable.Select(f => f.Box).ToList();

        // Pick the face closest to the image center (better than max-score when a
        // background face scores higher than the subject).
        float cx = width / 2f, cy = height / 2f;
        var chosen = usable
            .OrderBy(f => MathF.Pow((f.Box.CenterX - cx) / width, 2)
                        + MathF.Pow((f.Box.CenterY - cy) / height, 2))
            .First();
        FaceBox best = chosen.Box;

        log?.WriteLine($"        chosen        ({best.X},{best.Y}) {best.Width}x{best.Height}  score {best.Score:F3}");

        // Align before embedding � SFace expects the canonical 112x112 crop.
        // Hand the recognizer the detector's own row: it carries the sub-pixel
        // box and landmark values that FaceBox has already rounded to int.
        using (Mat subjectRow = _faces.Row(chosen.Row))
            _recognizer.AlignCrop(_image, subjectRow, _alignedFace);
        _recognizer.Feature(_alignedFace, _embedding);

        float[] embedding = new float[_embedding.Total];
        _embedding.CopyTo(embedding);

        return new FaceRecord(Path.GetFileName(imagePath), Path.GetFullPath(imagePath),
                              width, height, sharpness,
                              boxes, new Subject(best, best.Score), embedding);
    }

    // Every YuNet detection row is box(4) + landmarks(10) + score(1), packed as
    // contiguous float32, so one bulk copy per row beats per-cell reads.
    private const int RowValues = 15;

    private static List<FaceBox> ParseFaces(Mat faces)
    {
        var result = new List<FaceBox>();
        var v = new float[RowValues];
        for (int i = 0; i < faces.Rows; i++)
        {
            using Mat row = faces.Row(i);
            Marshal.Copy(row.DataPointer, v, 0, v.Length);
            result.Add(new FaceBox((int)v[0], (int)v[1], (int)v[2], (int)v[3], v[14],
                v[4], v[5],      // right eye
                v[6], v[7],      // left eye
                v[8], v[9],      // nose tip
                v[10], v[11],    // right mouth corner
                v[12], v[13]));  // left mouth corner
        }
        return result;
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _detector.Dispose();
        _recognizer.Dispose();
        _image.Dispose(); _faces.Dispose(); _alignedFace.Dispose(); _embedding.Dispose();
    }
}
