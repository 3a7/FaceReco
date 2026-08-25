using FaceID.App;

namespace FaceID.Tests;

/// <summary>
/// Locates the repository from the test binary's folder, so the sample images
/// and the bundled models can be reached no matter where the runner puts us.
/// </summary>
internal static class TestData
{
    public static string RepoRoot { get; } = FindRepoRoot();

    public static string DetectorModel =>
        Path.Combine(RepoRoot, "FaceID.App", "models", "face_detection_yunet_2026may.onnx");

    public static string RecognizerModel =>
        Path.Combine(RepoRoot, "FaceID.App", "models", "face_recognition_sface_2021dec.onnx");

    public static string ImageFolder => Path.Combine(RepoRoot, "imgs");

    public static string Image(string fileName) => Path.Combine(ImageFolder, fileName);

    private static string FindRepoRoot()
    {
        DirectoryInfo? dir = new(AppContext.BaseDirectory);
        while (dir is not null && !File.Exists(Path.Combine(dir.FullName, "FaceID.sln")))
            dir = dir.Parent;

        return dir?.FullName
            ?? throw new InvalidOperationException(
                $"could not find FaceID.sln walking up from {AppContext.BaseDirectory}");
    }
}

/// <summary>
/// Loading SFace costs about a second, so the engine is built once and shared
/// by every test in a collection rather than per test.
/// </summary>
public sealed class EngineFixture : IDisposable
{
    public FaceEngine Engine { get; }

    public EngineFixture() =>
        Engine = new FaceEngine(TestData.DetectorModel, TestData.RecognizerModel, 0.6f);

    public void Dispose() => Engine.Dispose();
}

[CollectionDefinition("engine")]
public sealed class EngineCollection : ICollectionFixture<EngineFixture>;
