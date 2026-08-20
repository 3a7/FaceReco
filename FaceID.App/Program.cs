using System.Diagnostics;

namespace FaceID.App;

public static class Program
{
    private const string DefaultGallery = "gallery.json";

    public static int Main(string[] args)
    {
        var options = new Options();
        List<string> positional;

        try
        {
            positional = options.Parse(args);
        }
        catch (ArgumentException ex)
        {
            Console.Error.WriteLine($"faceid: {ex.Message}");
            Console.Error.WriteLine("try 'faceid --help'");
            return 2;
        }

        if (options.Help || positional.Count == 0)
        {
            PrintUsage();
            return positional.Count == 0 && !options.Help ? 2 : 0;
        }

        string command = positional[0].ToLowerInvariant();
        List<string> rest = positional.Skip(1).ToList();

        try
        {
            return command switch
            {
                "recognize" => Recognize(rest, options),
                "compare"   => Compare(rest, options),
                "search"    => Search(rest, options),
                _ => Usage($"unknown command '{command}'"),
            };
        }
        catch (Exception ex) when (ex is IOException or InvalidDataException or InvalidOperationException)
        {
            Console.Error.WriteLine($"faceid: {ex.Message}");
            return 1;
        }
    }

    // ---------------------------------------------------------------- recognize

    private static int Recognize(List<string> args, Options options)
    {
        if (args.Count != 1) return Usage("recognize takes exactly one folder");

        string folder = args[0];
        if (!Directory.Exists(folder))
            return Fail($"no such folder: {folder}");

        string[] images = ImageFilesIn(folder);
        if (images.Length == 0) return Fail($"no images in {folder}");

        Console.WriteLine($"Scanning {images.Length} images in {Path.GetFullPath(folder)}");
        Console.WriteLine($"Detection threshold {options.DetectThreshold:F2}");
        Console.WriteLine();

        using var engine = new FaceEngine(options.DetectorModel, options.RecognizerModel, options.DetectThreshold);
        var stopwatch = Stopwatch.StartNew();

        var gallery = new Gallery
        {
            SourceFolder = Path.GetFullPath(folder),
            DetectThreshold = options.DetectThreshold,
        };

        int skipped = 0, crowded = 0, faces = 0;
        TextWriter? log = options.Quiet ? null : Console.Out;

        foreach (string path in images)
        {
            if (!options.Quiet) Console.WriteLine($"    {Path.GetFileName(path)}");

            FaceRecord? record = engine.Analyze(path, log);
            if (record is null)
            {
                if (options.Quiet) Console.WriteLine($"    {Path.GetFileName(path)}  - no face");
                skipped++;
                continue;
            }

            faces += record.FaceCount;
            if (record.FaceCount > 1) crowded++;

            gallery.Entries.Add(new GalleryEntry
            {
                File = record.File,
                Path = record.Path,
                Score = record.Subject.Score,
                FaceCount = record.FaceCount,
                Embedding = record.Embedding,
            });
        }

        stopwatch.Stop();

        Console.WriteLine();
        Console.WriteLine(new string('-', 60));
        Console.WriteLine($"{images.Length} images scanned in {stopwatch.Elapsed.TotalSeconds:F1}s");
        Console.WriteLine($"{faces} faces detected");
        Console.WriteLine($"{gallery.Entries.Count} identities enrolled");
        Console.WriteLine($"{skipped} images with no usable face");
        Console.WriteLine($"{crowded} images with more than one face");

        if (gallery.Entries.Count > 0)
        {
            GalleryEntry weakest = gallery.Entries.MinBy(e => e.Score)!;
            GalleryEntry strongest = gallery.Entries.MaxBy(e => e.Score)!;
            Console.WriteLine($"weakest subject  {weakest.File} at {weakest.Score:F3}");
            Console.WriteLine($"strongest subject {strongest.File} at {strongest.Score:F3}");
        }

        if (!options.NoSave)
        {
            gallery.Save(options.GalleryPath);
            Console.WriteLine($"gallery written to {Path.GetFullPath(options.GalleryPath)}");
        }

        return 0;
    }

    // ------------------------------------------------------------------ compare

    private static int Compare(List<string> args, Options options)
    {
        if (args.Count != 2) return Usage("compare takes exactly two images or identities");

        Gallery? gallery = TryLoadGallery(options.GalleryPath);

        // The models are only loaded if an argument turns out to be a file on
        // disk. Comparing two already-enrolled identities never touches them.
        FaceEngine? engine = null;
        try
        {
            float[]? a = Resolve(args[0], gallery, options, ref engine, out string labelA);
            if (a is null) return Fail($"no face for {args[0]}");
            float[]? b = Resolve(args[1], gallery, options, ref engine, out string labelB);
            if (b is null) return Fail($"no face for {args[1]}");

            double score = Similarity.Cosine(a, b);
            bool same = score >= options.MatchThreshold;

            Console.WriteLine();
            Console.WriteLine($"{labelA}");
            Console.WriteLine($"{labelB}");
            Console.WriteLine($"    cosine {score:F3} {(same ? ">=" : " <")} {options.MatchThreshold:F3}");
            Console.WriteLine($"    ->  {(same ? "SAME PERSON" : "NOT THE SAME PERSON")}");
            return 0;
        }
        finally { engine?.Dispose(); }
    }

    // ------------------------------------------------------------------- search

    private static int Search(List<string> args, Options options)
    {
        if (args.Count is < 1 or > 2)
            return Usage("search takes a probe image and optionally a folder");

        Gallery gallery;
        if (args.Count == 2)
        {
            string folder = args[1];
            if (!Directory.Exists(folder)) return Fail($"no such folder: {folder}");
            Console.WriteLine($"Scanning {folder} (no gallery given)");
            gallery = ScanQuietly(folder, options);
        }
        else
        {
            gallery = TryLoadGallery(options.GalleryPath)
                ?? throw new FileNotFoundException(
                    $"no gallery at {options.GalleryPath}. Run 'faceid recognize <folder>' first, "
                    + "or pass a folder as the second argument.");
        }

        if (gallery.Entries.Count == 0) return Fail("the gallery is empty");

        FaceEngine? engine = null;
        float[]? probe;
        string label;
        try
        {
            probe = Resolve(args[0], gallery, options, ref engine, out label);
        }
        finally { engine?.Dispose(); }

        if (probe is null) return Fail($"no face for {args[0]}");

        string probeName = Path.GetFileName(args[0]);
        var ranked = gallery.Entries
            .Where(e => !string.Equals(e.File, probeName, StringComparison.OrdinalIgnoreCase))
            .Select(e => (e.File, Score: Similarity.Cosine(probe, e.Embedding)))
            .OrderByDescending(r => r.Score)
            .ToList();

        var matched  = ranked.Where(r => r.Score >= options.MatchThreshold).ToList();
        var rejected = ranked.Where(r => r.Score <  options.MatchThreshold).ToList();

        Console.WriteLine();
        Console.WriteLine($"Looking for {label}");
        Console.WriteLine($"among {ranked.Count} enrolled identities");
        Console.WriteLine();

        if (matched.Count == 0)
            Console.WriteLine("    no match anywhere");
        foreach (var m in matched)
            Console.WriteLine($"    MATCH      {m.File,-34} {m.Score:F3}");

        Console.WriteLine($"    - - - - -  threshold {options.MatchThreshold:F3}  - - - - -");
        foreach (var r in rejected.Take(options.Top))
            Console.WriteLine($"    rejected   {r.File,-34} {r.Score:F3}");

        Console.WriteLine();
        Console.Write($"{matched.Count} match(es)");
        if (matched.Count > 0 && rejected.Count > 0)
            Console.Write($", clear of the closest rejection by {matched[^1].Score - rejected[0].Score:F3}");
        Console.WriteLine();
        return 0;
    }

    // ------------------------------------------------------------------ helpers

    /// <summary>
    /// An argument may be a path to an image on disk or the name of an already
    /// enrolled identity. A real file wins, so a fresh photo is always re-read.
    /// </summary>
    private static float[]? Resolve(
        string arg, Gallery? gallery, Options options, ref FaceEngine? engine, out string label)
    {
        if (File.Exists(arg))
        {
            label = $"{Path.GetFileName(arg)}  (read from disk)";
            engine ??= new FaceEngine(options.DetectorModel, options.RecognizerModel, options.DetectThreshold);
            FaceRecord? record = engine.Analyze(arg);
            return record?.Embedding;
        }

        GalleryEntry? entry = gallery?.Find(arg);
        if (entry is not null)
        {
            label = $"{entry.File}  (from gallery)";
            return entry.Embedding;
        }

        label = arg;
        return null;
    }

    private static Gallery ScanQuietly(string folder, Options options)
    {
        using var engine = new FaceEngine(options.DetectorModel, options.RecognizerModel, options.DetectThreshold);
        var gallery = new Gallery
        {
            SourceFolder = Path.GetFullPath(folder),
            DetectThreshold = options.DetectThreshold,
        };

        foreach (string path in ImageFilesIn(folder))
        {
            FaceRecord? record = engine.Analyze(path);
            if (record is null) continue;
            gallery.Entries.Add(new GalleryEntry
            {
                File = record.File,
                Path = record.Path,
                Score = record.Subject.Score,
                FaceCount = record.FaceCount,
                Embedding = record.Embedding,
            });
        }
        return gallery;
    }

    private static Gallery? TryLoadGallery(string path) =>
        File.Exists(path) ? Gallery.Load(path) : null;

    private static string[] ImageFilesIn(string folder) =>
        Directory.EnumerateFiles(folder)
                 .Where(p => Path.GetExtension(p).ToLowerInvariant()
                     is ".jpg" or ".jpeg" or ".png" or ".bmp")
                 .OrderBy(p => p, StringComparer.OrdinalIgnoreCase)
                 .ToArray();

    private static int Usage(string message)
    {
        Console.Error.WriteLine($"faceid: {message}");
        Console.Error.WriteLine("try 'faceid --help'");
        return 2;
    }

    private static int Fail(string message)
    {
        Console.Error.WriteLine($"faceid: {message}");
        return 1;
    }

    private static void PrintUsage()
    {
        Console.WriteLine("""
            faceid - detect, compare and search faces

            USAGE
              faceid recognize <folder>              scan a folder and enrol one identity per image
              faceid compare   <a> <b>               compare two images, or two enrolled identities
              faceid search    <probe> [folder]      find a probe face among the enrolled ones

            An identity is the 128 numbers SFace produces for a face. 'recognize' writes
            them to a gallery file so later commands do not have to scan again. Arguments
            to 'compare' and 'search' may be image paths or names already in the gallery.

            OPTIONS
              --gallery <path>      gallery file to read or write (default: gallery.json)
              --no-save             recognize: report results without writing the gallery
              --detect <float>      detection confidence floor, 0 to 1 (default: 0.60)
              --match <float>       same-person cosine cutoff (default: 0.371)
              --top <n>             search: how many near misses to list (default: 3)
              -q, --quiet           less per-image detail
              --detector <path>     override the YuNet model file
              --recognizer <path>   override the SFace model file
              -h, --help            this text

            EXAMPLES
              faceid recognize imgs
              faceid compare imgs/Abdullah_Gul_0003.jpg imgs/Abdullah_Gul_0004.jpg
              faceid compare Abdullah_Gul_0003.jpg Adam_Sandler_0003.jpg
              faceid search grafik.png imgs
              faceid search Adrien_Brody_0006.jpg --match 0.42

            EXIT CODES
              0 success    1 runtime error    2 bad usage
            """);
    }
}

/// <summary>Hand-rolled option parsing, so the tool carries no extra dependency.</summary>
internal sealed class Options
{
    public string GalleryPath { get; private set; } = "gallery.json";
    public float DetectThreshold { get; private set; } = 0.60f;
    public double MatchThreshold { get; private set; } = 0.371;
    public int Top { get; private set; } = 3;
    public bool Quiet { get; private set; }
    public bool NoSave { get; private set; }
    public bool Help { get; private set; }

    public string DetectorModel { get; private set; } =
        Path.Combine(AppContext.BaseDirectory, "models", "face_detection_yunet_2026may.onnx");
    public string RecognizerModel { get; private set; } =
        Path.Combine(AppContext.BaseDirectory, "models", "face_recognition_sface_2021dec.onnx");

    /// <summary>Consumes the options and returns whatever positional arguments remain.</summary>
    public List<string> Parse(string[] args)
    {
        var positional = new List<string>();

        for (int i = 0; i < args.Length; i++)
        {
            string arg = args[i];
            switch (arg)
            {
                case "-h" or "--help":    Help = true; break;
                case "-q" or "--quiet":   Quiet = true; break;
                case "--no-save":         NoSave = true; break;
                case "--gallery":         GalleryPath = Next(args, ref i, arg); break;
                case "--detector":        DetectorModel = Next(args, ref i, arg); break;
                case "--recognizer":      RecognizerModel = Next(args, ref i, arg); break;
                case "--detect":          DetectThreshold = ParseUnit(Next(args, ref i, arg), arg); break;
                case "--match":           MatchThreshold = ParseCosine(Next(args, ref i, arg), arg); break;
                case "--top":             Top = ParseCount(Next(args, ref i, arg), arg); break;
                default:
                    if (arg.StartsWith('-') && arg.Length > 1)
                        throw new ArgumentException($"unknown option '{arg}'");
                    positional.Add(arg);
                    break;
            }
        }
        return positional;
    }

    private static string Next(string[] args, ref int i, string option)
    {
        if (i + 1 >= args.Length) throw new ArgumentException($"{option} needs a value");
        return args[++i];
    }

    private static float ParseUnit(string text, string option)
    {
        if (!float.TryParse(text, System.Globalization.NumberStyles.Float,
                System.Globalization.CultureInfo.InvariantCulture, out float value))
            throw new ArgumentException($"{option} expects a number, got '{text}'");
        if (value is < 0 or > 1)
            throw new ArgumentException($"{option} must be between 0 and 1, got {text}");
        return value;
    }

    private static double ParseCosine(string text, string option)
    {
        if (!double.TryParse(text, System.Globalization.NumberStyles.Float,
                System.Globalization.CultureInfo.InvariantCulture, out double value))
            throw new ArgumentException($"{option} expects a number, got '{text}'");
        if (value is < -1 or > 1)
            throw new ArgumentException($"{option} must be between -1 and 1, got {text}");
        return value;
    }

    private static int ParseCount(string text, string option)
    {
        if (!int.TryParse(text, out int value) || value < 0)
            throw new ArgumentException($"{option} expects a non-negative whole number, got '{text}'");
        return value;
    }
}
