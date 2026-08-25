// Gallery.cs
//
// A persistent set of enrolled faces: file metadata + the SFace embedding vector.
// Stored as JSON so it is easy to inspect or post-process externally.
//
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace FaceTool;

public static class Similarity
{
    /// <summary>Cosine similarity between two equal-length vectors. Range [-1, 1].</summary>
    public static float Cosine(float[] a, float[] b)
    {
        if (a.Length != b.Length || a.Length == 0)
            throw new ArgumentException("vectors must be same non-zero length");
        double dot = 0, na = 0, nb = 0;
        for (int i = 0; i < a.Length; i++) { dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; }
        return (float)(dot / (Math.Sqrt(na) * Math.Sqrt(nb)));
    }
}

public sealed class GalleryEntry
{
    public required string File { get; set; }
    public required string Path { get; set; }
    public float Score { get; set; }
    public int FaceCount { get; set; }
    public required float[] Embedding { get; set; }

    // ---- NEW fields (v2) -------------------------------------------------
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public string? Person { get; set; }                 // optional identity name
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public double? Sharpness { get; set; }              // blur metric at enroll time
    public DateTimeOffset Enrolled { get; set; } = DateTimeOffset.Now;
}

public sealed class Gallery
{
    public int Version { get; set; } = 2;               // bumped from 1
    public List<GalleryEntry> Entries { get; set; } = [];

    private static readonly JsonSerializerOptions Options = new()
    {
        WriteIndented = true,
        NumberHandling = JsonNumberHandling.AllowNamedFloatingPointLiterals,
    };

    public void Save(string path)
    {
        string json = JsonSerializer.Serialize(this, Options);
        File.WriteAllText(path, json);
    }

    public static Gallery Load(string path)
    {
        Gallery g = JsonSerializer.Deserialize<Gallery>(File.ReadAllText(path), Options)
                    ?? throw new InvalidDataException($"could not parse gallery: {path}");
        if (g.Version > 2)
            throw new NotSupportedException(
                $"gallery version {g.Version} is newer than this tool supports (max 2)");
        // v1 galleries load fine: Person/Sharpness simply stay null.
        return g;
    }

    // ---- NEW query helpers ------------------------------------------------

    /// <summary>All embeddings belonging to one person.</summary>
    public IEnumerable<GalleryEntry> For(string person) =>
        Entries.Where(e => string.Equals(e.Person, person, StringComparison.OrdinalIgnoreCase));

    public IEnumerable<string> Persons =>
        Entries.Where(e => e.Person is not null).Select(e => e.Person!).Distinct(StringComparer.OrdinalIgnoreCase);

    /// <summary>Best cosine any embedding of 'person' achieves against 'probe' (max-pooling).</summary>
    public double BestMatch(float[] probe, string person) =>
        For(person).Select(e => (double)Similarity.Cosine(probe, e.Embedding))
                   .DefaultIfEmpty(-2).Max();

    public bool HasNamedPersons() => Persons.Any();
}
