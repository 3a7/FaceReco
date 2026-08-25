using FaceID.App;

namespace FaceID.Tests;

public class GalleryTests : IDisposable
{
    private readonly string _dir = Directory.CreateTempSubdirectory("faceid-tests-").FullName;

    private string TempFile(string name) => Path.Combine(_dir, name);

    public void Dispose()
    {
        try { Directory.Delete(_dir, recursive: true); } catch (IOException) { }
        GC.SuppressFinalize(this);
    }

    private static Gallery SampleGallery() => new()
    {
        SourceFolder = @"C:\somewhere\imgs",
        DetectThreshold = 0.6f,
        Entries =
        [
            new GalleryEntry
            {
                File = "Abdullah_Gul_0003.jpg",
                Path = @"C:\somewhere\imgs\Abdullah_Gul_0003.jpg",
                Score = 0.912f,
                FaceCount = 1,
                Embedding = [0.5f, -1.25f, 3.75f],
            },
            new GalleryEntry
            {
                File = "Adam_Sandler_0003.jpg",
                Path = @"C:\somewhere\imgs\Adam_Sandler_0003.jpg",
                Score = 0.874f,
                FaceCount = 2,
                Embedding = [-2f, 0f, 1f],
            },
        ],
    };

    [Fact]
    public void SaveThenLoad_RoundTripsEveryField()
    {
        string path = TempFile("gallery.json");
        Gallery original = SampleGallery();
        original.Save(path);

        Gallery loaded = Gallery.Load(path);

        Assert.Equal(original.Version, loaded.Version);
        Assert.Equal(original.SourceFolder, loaded.SourceFolder);
        Assert.Equal(original.DetectThreshold, loaded.DetectThreshold);
        Assert.Equal(original.Entries.Count, loaded.Entries.Count);

        for (int i = 0; i < original.Entries.Count; i++)
        {
            GalleryEntry expected = original.Entries[i], actual = loaded.Entries[i];
            Assert.Equal(expected.File, actual.File);
            Assert.Equal(expected.Path, actual.Path);
            Assert.Equal(expected.Score, actual.Score);
            Assert.Equal(expected.FaceCount, actual.FaceCount);
            Assert.Equal(expected.Embedding, actual.Embedding);
        }
    }

    /// <summary>
    /// Embeddings are the whole point of the file, so they must survive the JSON
    /// round trip bit for bit rather than approximately.
    /// </summary>
    [Fact]
    public void SaveThenLoad_PreservesEmbeddingsExactly()
    {
        var random = new Random(99);
        var embedding = new float[128];
        for (int i = 0; i < embedding.Length; i++)
            embedding[i] = (float)(random.NextDouble() * 30 - 15);

        string path = TempFile("exact.json");
        new Gallery { Entries = [new GalleryEntry { File = "a.jpg", Embedding = embedding }] }.Save(path);

        float[] loaded = Gallery.Load(path).Entries[0].Embedding;

        Assert.Equal(embedding.Length, loaded.Length);
        for (int i = 0; i < embedding.Length; i++)
            Assert.True(embedding[i] == loaded[i], $"index {i}: {embedding[i]} became {loaded[i]}");
    }

    [Fact]
    public void Save_CreatesMissingDirectories()
    {
        string path = TempFile(Path.Combine("nested", "deeper", "gallery.json"));
        SampleGallery().Save(path);

        Assert.True(File.Exists(path));
    }

    [Fact]
    public void Load_OnMissingFile_Throws()
    {
        Assert.Throws<FileNotFoundException>(() => Gallery.Load(TempFile("absent.json")));
    }

    [Fact]
    public void Load_OnMalformedJson_Throws()
    {
        string path = TempFile("broken.json");
        File.WriteAllText(path, "{ this is not json");

        Assert.ThrowsAny<Exception>(() => Gallery.Load(path));
    }

    [Fact]
    public void Load_OnUnsupportedVersion_Throws()
    {
        string path = TempFile("future.json");
        File.WriteAllText(path, """{"Version":99,"Entries":[]}""");

        var error = Assert.Throws<InvalidDataException>(() => Gallery.Load(path));
        Assert.Contains("99", error.Message);
    }

    [Fact]
    public void Find_ByFileName()
    {
        Assert.Equal("Abdullah_Gul_0003.jpg", SampleGallery().Find("Abdullah_Gul_0003.jpg")?.File);
    }

    [Fact]
    public void Find_ByFullPath()
    {
        Gallery gallery = SampleGallery();
        Assert.Equal("Adam_Sandler_0003.jpg",
            gallery.Find(@"C:\somewhere\imgs\Adam_Sandler_0003.jpg")?.File);
    }

    [Fact]
    public void Find_IsCaseInsensitive()
    {
        Assert.NotNull(SampleGallery().Find("abdullah_gul_0003.JPG"));
    }

    /// <summary>
    /// Lets a caller pass a path that does not exist on this machine and still
    /// hit the entry, which is what makes a gallery portable between folders.
    /// </summary>
    [Fact]
    public void Find_FallsBackToTheFileNamePartOfAPath()
    {
        Assert.Equal("Abdullah_Gul_0003.jpg",
            SampleGallery().Find(@"D:\elsewhere\Abdullah_Gul_0003.jpg")?.File);
    }

    [Fact]
    public void Find_ReturnsNullForSomethingNotEnrolled()
    {
        Assert.Null(SampleGallery().Find("Nobody_9999.jpg"));
    }

    [Fact]
    public void NewGallery_IsVersionOneAndEmpty()
    {
        var gallery = new Gallery();
        Assert.Equal(1, gallery.Version);
        Assert.Empty(gallery.Entries);
    }
}
