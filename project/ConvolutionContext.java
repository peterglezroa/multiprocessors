/*----------------------------------------------------------------
* Programación avanzada: Proyecto final
* Fecha: 25-Nov-2021
* Autor: A01651517 Pedro González
*--------------------------------------------------------------*/
import java.awt.image.BufferedImage;
import java.io.File;
import javax.imageio.ImageIO;
import java.io.IOException;

public class ConvolutionContext {
    private BufferedImage source, destination;
	private int src[], dest[];
    private float kernel[];

    public ConvolutionContext(String fileName, boolean color) {
		File srcFile = new File(fileName);
        source = ImageIO.read(srcFile);
    }

    public int getSize() {
        return source.getWidth()*source.getHeight();
    }

    public int[] getData() {
        return source.getRGB(0, 0, source.getWidth(), source.getHeight(), null,
            0 source.getWidth());
    }
}
