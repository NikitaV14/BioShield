import com.machinezoo.sourceafis.*;
import java.nio.file.*;

public class FingerprintBridge {
    public static void main(String[] args) throws Exception {
        if (args.length < 2) {
            System.err.println("Usage: FingerprintBridge <mode> <image1> [image2]");
            System.exit(1);
        }
        String mode = args[0];

        if (mode.equals("score")) {
            byte[] img1 = Files.readAllBytes(Paths.get(args[1]));
            byte[] img2 = Files.readAllBytes(Paths.get(args[2]));
            FingerprintTemplate probe = new FingerprintTemplate(
                new FingerprintImage(img1, new FingerprintImageOptions().dpi(500)));
            FingerprintTemplate candidate = new FingerprintTemplate(
                new FingerprintImage(img2, new FingerprintImageOptions().dpi(500)));
            double score = new FingerprintMatcher(probe).match(candidate);
            System.out.println(score);

        } else if (mode.equals("template")) {
            byte[] img = Files.readAllBytes(Paths.get(args[1]));
            FingerprintTemplate template = new FingerprintTemplate(
                new FingerprintImage(img, new FingerprintImageOptions().dpi(500)));
            byte[] serialized = template.toByteArray();
            StringBuilder sb = new StringBuilder();
            for (int i = 0; i < serialized.length; i++) {
                sb.append(serialized[i] & 0xFF);
                if (i < serialized.length - 1) sb.append(",");
            }
            System.out.println(sb.toString());
        }
    }
}
