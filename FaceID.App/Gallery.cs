using System.Text.Json;
using System.Text.Json.Serialization;

namespace FaceID.App;

/// <summary>One enrolled face. The embedding is the identity; everything else is provenance.</summary>
public sealed class GalleryEntry
{
    public string File { get; set; } = "";
    public string Path { get; set; } = "";
    public float Score { get; set; }
    public int FaceCount { get; set; }
    public float[] Embedding { get; set; } = [];
}

/// <summary>
/// A set of enrolled identities written to disk, so a scan is paid for once and
/// later comparisons and searches load in milliseconds.
/// </summary>
public sealed class Gallery
{
    public int Version { get; set; } = 1;
    public DateTimeOffset Created { get; set; } = DateTimeOffset.Now;
    public string SourceFolder { get; set; } = "";
    public float DetectThreshold { get; set; }
    public List<GalleryEntry> Entries { get; set; } = [];

    private static readonly JsonSerializerOptions Options = new()
    {
        WriteIndented = false,
        DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
    };

    public void Save(string path)
    {
        string? dir = System.IO.Path.GetDirectoryName(System.IO.Path.GetFullPath(path));
        if (!string.IsNullOrEmpty(dir)) Directory.CreateDirectory(dir);
        using FileStream stream = File.Create(path);
        JsonSerializer.Serialize(stream, this, Options);
    }

    public static Gallery Load(string path)
    {
        if (!File.Exists(path))
            throw new FileNotFoundException($"gallery not found: {path}");

        using FileStream stream = File.OpenRead(path);
        Gallery? gallery = JsonSerializer.Deserialize<Gallery>(stream, Options)
            ?? throw new InvalidDataException($"gallery is empty or malformed: {path}");

        if (gallery.Version != 1)
            throw new InvalidDataException($"unsupported gallery version {gallery.Version} in {path}");

        return gallery;
    }

    /// <summary>Looks an entry up by file name or by full path, case-insensitively.</summary>
    public GalleryEntry? Find(string nameOrPath)
    {
        return Entries.FirstOrDefault(e =>
                   string.Equals(e.File, nameOrPath, StringComparison.OrdinalIgnoreCase))
            ?? Entries.FirstOrDefault(e =>
                   string.Equals(e.Path, nameOrPath, StringComparison.OrdinalIgnoreCase))
            ?? Entries.FirstOrDefault(e =>
                   string.Equals(e.File, System.IO.Path.GetFileName(nameOrPath), StringComparison.OrdinalIgnoreCase));
    }
}
