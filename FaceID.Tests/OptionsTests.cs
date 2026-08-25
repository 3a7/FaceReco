using System.Globalization;
using FaceID.App;

namespace FaceID.Tests;

public class OptionsTests
{
    [Fact]
    public void Defaults_AreTheDocumentedOnes()
    {
        var options = new Options();
        options.Parse([]);

        Assert.Equal("gallery.json", options.GalleryPath);
        Assert.Equal(0.60f, options.DetectThreshold);
        Assert.Equal(0.371, options.MatchThreshold);
        Assert.Equal(3, options.Top);
        Assert.False(options.Quiet);
        Assert.False(options.NoSave);
        Assert.False(options.Help);
    }

    [Fact]
    public void ModelPaths_DefaultToTheBundledModels()
    {
        var options = new Options();
        options.Parse([]);

        Assert.EndsWith("face_detection_yunet_2026may.onnx", options.DetectorModel);
        Assert.EndsWith("face_recognition_sface_2021dec.onnx", options.RecognizerModel);
        Assert.True(Path.IsPathRooted(options.DetectorModel));
    }

    [Fact]
    public void PositionalArguments_ComeBackInOrder()
    {
        var options = new Options();
        List<string> positional = options.Parse(["compare", "a.jpg", "b.jpg"]);

        Assert.Equal(["compare", "a.jpg", "b.jpg"], positional);
    }

    [Fact]
    public void OptionsAndPositionals_CanBeInterleaved()
    {
        var options = new Options();
        List<string> positional = options.Parse(["search", "--match", "0.5", "probe.jpg", "-q"]);

        Assert.Equal(["search", "probe.jpg"], positional);
        Assert.Equal(0.5, options.MatchThreshold);
        Assert.True(options.Quiet);
    }

    [Theory]
    [InlineData("-q")]
    [InlineData("--quiet")]
    public void QuietFlag_HasBothSpellings(string flag)
    {
        var options = new Options();
        options.Parse([flag]);
        Assert.True(options.Quiet);
    }

    [Theory]
    [InlineData("-h")]
    [InlineData("--help")]
    public void HelpFlag_HasBothSpellings(string flag)
    {
        var options = new Options();
        options.Parse([flag]);
        Assert.True(options.Help);
    }

    [Fact]
    public void ValueOptions_AreRead()
    {
        var options = new Options();
        options.Parse(
        [
            "--gallery", "out/faces.json",
            "--detect", "0.8",
            "--match", "0.42",
            "--top", "7",
            "--no-save",
            "--detector", "d.onnx",
            "--recognizer", "r.onnx",
        ]);

        Assert.Equal("out/faces.json", options.GalleryPath);
        Assert.Equal(0.8f, options.DetectThreshold);
        Assert.Equal(0.42, options.MatchThreshold);
        Assert.Equal(7, options.Top);
        Assert.True(options.NoSave);
        Assert.Equal("d.onnx", options.DetectorModel);
        Assert.Equal("r.onnx", options.RecognizerModel);
    }

    /// <summary>
    /// This machine runs a locale where the decimal separator is a comma. Parsing
    /// has to be culture invariant or "0.42" would be read as 42 on the command
    /// line while still looking correct in a test written on an English machine.
    /// </summary>
    [Fact]
    public void DecimalValues_ParseRegardlessOfLocale()
    {
        CultureInfo original = CultureInfo.CurrentCulture;
        try
        {
            CultureInfo.CurrentCulture = new CultureInfo("de-DE");

            var options = new Options();
            options.Parse(["--match", "0.42", "--detect", "0.75"]);

            Assert.Equal(0.42, options.MatchThreshold);
            Assert.Equal(0.75f, options.DetectThreshold);
        }
        finally
        {
            CultureInfo.CurrentCulture = original;
        }
    }

    [Fact]
    public void UnknownOption_Throws()
    {
        var options = new Options();
        var error = Assert.Throws<ArgumentException>(() => options.Parse(["--nope"]));
        Assert.Contains("--nope", error.Message);
    }

    [Fact]
    public void OptionMissingItsValue_Throws()
    {
        var options = new Options();
        var error = Assert.Throws<ArgumentException>(() => options.Parse(["--match"]));
        Assert.Contains("--match", error.Message);
    }

    [Theory]
    [InlineData("--detect", "abc")]
    [InlineData("--detect", "-0.1")]
    [InlineData("--detect", "1.5")]
    [InlineData("--match", "abc")]
    [InlineData("--match", "2")]
    [InlineData("--match", "-1.5")]
    [InlineData("--top", "-1")]
    [InlineData("--top", "half")]
    public void OutOfRangeOrUnparseableValues_Throw(string option, string value)
    {
        var options = new Options();
        Assert.Throws<ArgumentException>(() => options.Parse([option, value]));
    }

    [Theory]
    [InlineData("--match", "-1")]
    [InlineData("--match", "1")]
    [InlineData("--detect", "0")]
    [InlineData("--detect", "1")]
    [InlineData("--top", "0")]
    public void BoundaryValues_AreAccepted(string option, string value)
    {
        var options = new Options();
        options.Parse([option, value]);
    }

    /// <summary>A bare "-" is a filename, not an option, and must not be rejected.</summary>
    [Fact]
    public void SingleDash_IsTreatedAsPositional()
    {
        var options = new Options();
        Assert.Equal(["-"], options.Parse(["-"]));
    }

    [Fact]
    public void NegativeLookingPositional_IsStillRejectedAsAnOption()
    {
        // Guards the current behaviour: anything starting with '-' that is longer
        // than one character is an option, so a file literally named "-x.jpg"
        // has to be passed as "./-x.jpg".
        var options = new Options();
        Assert.Throws<ArgumentException>(() => options.Parse(["-x.jpg"]));
    }
}
