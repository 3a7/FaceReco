using System.Drawing;
using System.Runtime.InteropServices;
using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Dnn;

namespace FaceID.App;

/// <summary>One detected face: the box, the five landmarks and the confidence.</summary>
public sealed record FaceBox(float X, float Y, float Width, float Height, float Score, PointF[] Landmarks)
{
    public PointF Center => new(X + Width / 2, Y + Height / 2);
    public float Area => Width * Height;

    /// <summary>Degrees the eye line is off horizontal. Alignment cancels this out.</summary>
    public double EyeTilt =>
        Math.Atan2(Landmarks[1].Y - Landmarks[0].Y, Landmarks[1].X - Landmarks[0].X) * 180 / Math.PI;
}

/// <summary>The result of running all four stages over one image.</summary>
public sealed record FaceRecord(string File, string Path, FaceBox Subject, int FaceCount, float[] Embedding);

/// <summary>
/// The pipeline: capture, detect, align, embed. Both models are loaded once and
/// reused, because constructing them is far more expensive than running them.
/// </summary>
public sealed class FaceEngine : IDisposable
{
    // Every YuNet detection row is box(4) + landmarks(10) + score(1).
    private const int RowValues = 15;

    private readonly FaceDetectorYN _detector;
    private readonly FaceRecognizerSF _recognizer;

    public FaceEngine(string detectorModel, string recognizerModel, float scoreThreshold)
    {
        if (!System.IO.File.Exists(detectorModel))
            throw new FileNotFoundException($"detector model not found: {detectorModel}");
        if (!System.IO.File.Exists(recognizerModel))
            throw new FileNotFoundException($"recognizer model not found: {recognizerModel}");

        _detector = new FaceDetectorYN(detectorModel, "", new Size(320, 320),
            scoreThreshold, 0.3f, 5000, Emgu.CV.Dnn.Backend.Default, Target.Cpu);
        _recognizer = new FaceRecognizerSF(recognizerModel, "",
            Emgu.CV.Dnn.Backend.Default, Target.Cpu);
    }

    /// <summary>Retunes the confidence floor without rebuilding the detector.</summary>
    public void SetScoreThreshold(float threshold) => _detector.SetScoreThreshold(threshold);

    /// <summary>
    /// Runs stages 1 to 4 over one image. Returns null when the file cannot be
    /// read or no face clears the threshold. Pass a writer for the verbose trace.
    /// </summary>
    public FaceRecord? Analyze(string imagePath, TextWriter? log = null)
    {
        // STAGE 1 - capture. ColorBgr because both models were trained on
        // OpenCV's native blue-green-red order; ColorRgb silently lowers scores.
        using Mat image = CvInvoke.Imread(imagePath, ImreadModes.ColorBgr);
        if (image.IsEmpty)
        {
            log?.WriteLine("        could not read this file");
            return null;
        }

        log?.WriteLine($"        image         {image.Width}x{image.Height}, "
                     + $"{image.NumberOfChannels} channels, {image.Depth}");

        // STAGE 2 - detection. The detector must be told the image size before
        // every call, or it throws when the size differs from the preset one.
        _detector.InputSize = new Size(image.Width, image.Height);
        using var detections = new Mat();
        _detector.Detect(image, detections);   // returns 1 for "ran ok", not a count

        if (detections.Rows == 0)
        {
            log?.WriteLine("        no face cleared the threshold");
            return null;
        }

        var boxes = new FaceBox[detections.Rows];
        for (int i = 0; i < detections.Rows; i++)
            boxes[i] = ReadRow(detections, i);

        // One identity per file, so pick the most central face. Images often
        // contain bystanders, and the highest-scoring face is sometimes one.
        int subject = 0;
        double best = double.MaxValue;
        for (int i = 0; i < boxes.Length; i++)
        {
            double dx = boxes[i].Center.X - image.Width / 2.0;
            double dy = boxes[i].Center.Y - image.Height / 2.0;
            if (dx * dx + dy * dy < best) { best = dx * dx + dy * dy; subject = i; }
        }

        if (log is not null)
        {
            for (int i = 0; i < boxes.Length; i++)
            {
                FaceBox b = boxes[i];
                log.WriteLine($"        face ({i + 1})      score {b.Score:F3}  "
                            + $"box {b.X:F0},{b.Y:F0} {b.Width:F0}x{b.Height:F0}  area {b.Area:F0}");
            }
            if (boxes.Length > 1)
                log.WriteLine($"        subject       face ({subject + 1}), the most central one");
            log.WriteLine($"        eye tilt      {boxes[subject].EyeTilt:F2} deg");
        }

        using Mat subjectRow = detections.Row(subject);

        // STAGE 3 - alignment. The five landmarks drive a rotate-and-scale onto a
        // canonical 112x112 crop, so tilt, distance and framing stop mattering.
        using var aligned = new Mat();
        _recognizer.AlignCrop(image, subjectRow, aligned);
        log?.WriteLine($"        aligned       {aligned.Width}x{aligned.Height}, "
                     + $"{aligned.NumberOfChannels} channels, {aligned.Depth}");

        // STAGE 4 - embedding. SFace turns that crop into 128 numbers, and those
        // numbers are the identity.
        using var feature = new Mat();
        _recognizer.Feature(aligned, feature);

        var embedding = new float[feature.Cols];
        Marshal.Copy(feature.DataPointer, embedding, 0, embedding.Length);

        log?.WriteLine($"        embedded      {feature.Rows}x{feature.Cols} numbers, "
                     + $"vector length {Similarity.Length(embedding):F2}");

        return new FaceRecord(
            System.IO.Path.GetFileName(imagePath), imagePath, boxes[subject], boxes.Length, embedding);
    }

    private static FaceBox ReadRow(Mat detections, int index)
    {
        using Mat row = detections.Row(index);
        var v = new float[RowValues];
        Marshal.Copy(row.DataPointer, v, 0, v.Length);

        // Landmark sides are from the pictured person's point of view, so their
        // right eye appears on the left of the image.
        var landmarks = new[]
        {
            new PointF(v[4],  v[5]),    // right eye
            new PointF(v[6],  v[7]),    // left eye
            new PointF(v[8],  v[9]),    // nose tip
            new PointF(v[10], v[11]),   // right mouth corner
            new PointF(v[12], v[13]),   // left mouth corner
        };
        return new FaceBox(v[0], v[1], v[2], v[3], v[14], landmarks);
    }

    public void Dispose()
    {
        _detector.Dispose();
        _recognizer.Dispose();
    }
}

public static class Similarity
{
    /// <summary>
    /// SFace embeddings are NOT unit length, so both lengths must be divided out.
    /// Skip that and the result is not a cosine, and the threshold means nothing.
    /// </summary>
    public static double Cosine(float[] a, float[] b)
    {
        if (a.Length != b.Length)
            throw new ArgumentException($"embedding sizes differ: {a.Length} vs {b.Length}");

        double dot = 0, normA = 0, normB = 0;
        for (int i = 0; i < a.Length; i++)
        {
            dot   += a[i] * b[i];
            normA += a[i] * a[i];
            normB += b[i] * b[i];
        }
        return dot / (Math.Sqrt(normA) * Math.Sqrt(normB));
    }

    public static double Length(float[] v)
    {
        double sum = 0;
        foreach (float f in v) sum += (double)f * f;
        return Math.Sqrt(sum);
    }
}
