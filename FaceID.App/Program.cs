// Program.cs
//
// Commands:
//   scan      walk a folder, find faces, save gallery.json
//   search    probe one image against the saved gallery (file-level or person-level)
//   enroll    NEW  name + images/folders -> add named identity samples to the gallery
//   cluster   NEW  group all gallery entries into likely identities (union-find)
//   export    NEW  dump the gallery as CSV
//
// Options:
//   --threshold F   detection confidence cutoff        default 0.60
//   --match F       cosine similarity match cutoff     default 0.363
//   --gallery PATH  where to read/write the gallery    default gallery.json
//   --min-face N    NEW  smallest usable face, px      default 48
//   --gpu           NEW  run on CUDA instead of CPU
//
using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using Emgu.CV.CvEnum;
using Emgu.CV.Dnn;
using FaceTool;

namespace FaceTool;

internal static class Program
{
    private sealed class Options
    {
        public float Threshold = 0.60f;
        public float Match = 0.363f;
        public string GalleryPath = "gallery.json";
        public float MinFace = 48f;
        public bool Gpu = false;
    }

    private static readonly string[] ImageExtensions =
        [".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"];

    private static int Main(string[] args)
    {
        try
        {
            if (args.Length == 0) return Usage(null);

            string cmd = args[0].ToLowerInvariant();
            if (cmd is "-h" or "--help") return Usage(null);

            var options = new Options();
            var rest = new List<string>();
            for (int i = 1; i < args.Length; i++)
            {
                string arg = args[i];
                switch (arg)
                {
                    case "--threshold": options.Threshold = Next(args, ref i, arg); break;
                    case "--match": options.Match = Next(args, ref i, arg); break;
                    case "--gallery": options.GalleryPath = NextString(args, ref i, arg); break;
                    case "--min-face": options.MinFace = Next(args, ref i, arg); break;
                    case "--gpu": options.Gpu = true; break;
                    case "--": break; // everything after -- is positional
                    default:
                        if (arg.StartsWith("--")) throw new ArgumentException($"unknown option '{arg}'");
                        rest.Add(arg);
                        break;
                }
            }

            return cmd switch
            {
                "scan" => Scan(rest, options),
                "search" => Search(rest, options),
                "enroll" => Enroll(rest, options),
                "cluster" => Cluster(rest, options),
                "export" => Export(rest, options),
                _ => Usage($"unknown command '{cmd}'"),
            };
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"error: {ex.Message}");
            return 2;
        }
    }

    // ------------------------------------------------------------------ scan

    private static int Scan(List<string> args, Options options)
    {
        if (args.Count == 0) return Usage("scan takes one or more files/folders");

        var paths = args.SelectMany(a => Directory.Exists(a)
            ? ImageFilesIn(a)
            : File.Exists(a) ? [a]
            : []).Distinct().ToList();

        if (paths.Count == 0) return Fail("no readable images given");
        Console.WriteLine($"{paths.Count} image(s) to process\n");

        using var engine = MakeEngine(options);
        var gallery = new Gallery();

        foreach (string p in paths)
        {
            Console.WriteLine($"    {Path.GetFileName(p)}");
            FaceRecord? r = engine.Analyze(p, Console.Out);
            if (r is not null)
                gallery.Entries.Add(new GalleryEntry
                {
                    File = r.File,
                    Path = r.Path,
                    Score = r.Subject.Score,
                    FaceCount = r.Boxes.Count,
                    Embedding = r.Embedding,
                    Sharpness = r.Sharpness,   // NEW
                });
        }

        gallery.Save(options.GalleryPath);
        Console.WriteLine($"\n{gallery.Entries.Count}/{paths.Count} enrolled -> {options.GalleryPath}");
        return 0;
    }

    // ----------------------------------------------------------------- search

    private static int Search(List<string> args, Options options)
    {
        if (args.Count != 1) return Usage("search takes exactly one probe image");
        string probePath = args[0];
        if (!File.Exists(probePath)) return Fail($"no such file: {probePath}");

        Gallery gallery = TryLoadGallery(options.GalleryPath);
        if (gallery.Entries.Count == 0) return Fail("gallery is empty - run 'scan' first");

        using var engine = MakeEngine(options);
        FaceRecord? probe = engine.Analyze(probePath, Console.Out);
        if (probe is null) return Fail("no usable face in probe image");

        // NEW: person-level matching when identities exist (max-pooled over samples).
        if (gallery.HasNamedPersons())
        {
            Console.WriteLine("\nbest matches (per person):");
            foreach ((string person, double score) in gallery.Persons
                         .Select(p => (Person: p, gallery.BestMatch(probe.Embedding, p)))
                         .OrderByDescending(r => r.Item2).Take(5))
            {
                string mark = score >= options.Match ? "<-- MATCH" : "";
                Console.WriteLine($"    {score,7:F4}  {person,-24} {mark}");
            }
            return 0;
        }

        // Fall back to per-file ranking (original behavior).
        var ranked = gallery.Entries
            .Select(e => (Entry: e, Score: Similarity.Cosine(probe.Embedding, e.Embedding)))
            .OrderByDescending(r => r.Score)
            .Take(5);

        Console.WriteLine("\nbest matches:");
        foreach (var (entry, score) in ranked)
        {
            string mark = score >= options.Match ? "<-- MATCH" : "";
            Console.WriteLine($"    {score,7:F4}  {entry.File,-34} {mark}");
        }
        return 0;
    }

    // ----------------------------------------------------------------- enroll

    private static int Enroll(List<string> args, Options options)
    {
        if (args.Count < 2) return Usage("enroll takes a name then one or more images/folders");
        string person = args[0];

        var paths = args.Skip(1).SelectMany(a => Directory.Exists(a)
            ? ImageFilesIn(a)
            : File.Exists(a) ? [a]
            : []).Distinct().ToList();
        if (paths.Count == 0) return Fail("no readable images given");

        Gallery gallery = TryLoadGallery(options.GalleryPath) ?? new Gallery();
        using var engine = MakeEngine(options);

        int added = 0;
        foreach (string p in paths)
        {
            Console.WriteLine($"    {Path.GetFileName(p)}");
            FaceRecord? r = engine.Analyze(p, Console.Out);
            if (r is null) continue;

            // Duplicate guard: skip near-identical embeddings already stored.
            if (gallery.Entries.Any(e => Similarity.Cosine(e.Embedding, r.Embedding) > 0.99))
            {
                Console.WriteLine("        duplicate of existing entry, skipped");
                continue;
            }

            gallery.Entries.Add(new GalleryEntry
            {
                File = r.File,
                Path = r.Path,
                Score = r.Subject.Score,
                FaceCount = r.Boxes.Count,
                Embedding = r.Embedding,
                Person = person,        // NEW
                Sharpness = r.Sharpness,   // NEW
            });
            added++;
        }

        gallery.Save(options.GalleryPath);
        Console.WriteLine($"\n'{person}': {added} new sample(s), "
                        + $"{gallery.For(person).Count()} total -> {options.GalleryPath}");
        return 0;
    }

    // ---------------------------------------------------------------- cluster

    private static int Cluster(List<string> _, Options options)
    {
        Gallery g = TryLoadGallery(options.GalleryPath);
        if (g.Entries.Count == 0) return Fail("gallery is empty");

        int n = g.Entries.Count;
        var parent = Enumerable.Range(0, n).ToArray();
        int Find(int i) { while (parent[i] != i) i = parent[i] = parent[parent[i]]; return i; }

        // Union-find: link any pair above the match threshold.
        for (int i = 0; i < n; i++)
            for (int j = i + 1; j < n; j++)
                if (Similarity.Cosine(g.Entries[i].Embedding, g.Entries[j].Embedding)
                    >= options.Match)
                    parent[Find(i)] = Find(j);

        Console.WriteLine($"{n} entries grouped by cosine >= {options.Match:F3}\n");
        foreach (var group in Enumerable.Range(0, n).GroupBy(Find)
                                        .OrderByDescending(gr => gr.Count()))
        {
            Console.WriteLine($"cluster of {group.Count()}:");
            foreach (int i in group)
            {
                var e = g.Entries[i];
                string person = e.Person ?? "-";
                Console.WriteLine($"    {e.File,-34} {person,-20} score {e.Score:F3}");
            }
            Console.WriteLine();
        }
        return 0;
    }

    // ----------------------------------------------------------------- export

    private static int Export(List<string> args, Options options)
    {
        string outPath = args.Count == 1 ? args[0] : "gallery.csv";
        Gallery g = TryLoadGallery(options.GalleryPath);
        if (g.Entries.Count == 0) return Fail("gallery is empty");

        using var w = new StreamWriter(outPath);
        w.WriteLine("file,person,score,faces,sharpness,enrolled");
        foreach (var e in g.Entries)
            // Invariant throughout: a decimal comma would split one value across
            // two CSV columns on locales like de-DE.
            w.WriteLine($"\"{e.File}\",\"{e.Person ?? ""}\","
                      + $"{e.Score.ToString("F3", CultureInfo.InvariantCulture)},{e.FaceCount},"
                      + $"{e.Sharpness?.ToString("F0", CultureInfo.InvariantCulture) ?? ""},{e.Enrolled:O}");
        Console.WriteLine($"{g.Entries.Count} rows -> {outPath}");
        return 0;
    }

    // ------------------------------------------------------------------ misc

    private static FaceEngine MakeEngine(Options o) =>
        new(FaceEngine.DefaultDetectorModel, FaceEngine.DefaultRecognizerModel,
            o.Threshold, o.MinFace,
            o.Gpu ? Target.Cuda : Target.Cpu);

    private static Gallery TryLoadGallery(string path) =>
        File.Exists(path)
            ? Gallery.Load(path)
            : throw new FileNotFoundException($"no gallery at {path}", path);

    private static IEnumerable<string> ImageFilesIn(string dir) =>
        Directory.EnumerateFiles(dir, "*", SearchOption.AllDirectories)
                 .Where(f => ImageExtensions.Contains(Path.GetExtension(f).ToLowerInvariant()));

    private static float Next(IReadOnlyList<string> a, ref int i, string opt)
    {
        if (!float.TryParse(NextString(a, ref i, opt),
                            NumberStyles.Float, CultureInfo.InvariantCulture, out float v))
            throw new ArgumentException($"{opt} needs a number");
        return v;
    }

    private static string NextString(IReadOnlyList<string> a, ref int i, string opt)
    {
        if (++i >= a.Count) throw new ArgumentException($"{opt} needs a value");
        return a[i];
    }

    private static int Usage(string? error)
    {
        if (error is not null) Console.Error.WriteLine($"error: {error}\n");
        Console.WriteLine("""
            usage:
              FaceTool scan    <files-or-folders...> [--gallery P] [--threshold F] [--min-face N] [--gpu]
              FaceTool search  <probe-image>        [--gallery P] [--match F]
              FaceTool enroll  <name> <images...>   [--gallery P]
              FaceTool cluster                      [--gallery P] [--match F]
              FaceTool export  [out.csv]            [--gallery P]

            examples:
              FaceTool scan ./photos --gallery db.json
              FaceTool enroll alice ./alice-pics
              FaceTool enroll bob   bob1.jpg bob2.jpg
              FaceTool search probe.jpg --gallery db.json
              FaceTool cluster --gallery db.json
            """);
        return error is null ? 0 : 1;
    }

    private static int Fail(string message) { Console.Error.WriteLine($"error: {message}"); return 2; }
}
