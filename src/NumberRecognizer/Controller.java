package NumberRecognizer;

import javafx.embed.swing.SwingFXUtils;
import javafx.fxml.FXML;
import javafx.scene.SnapshotParameters;
import javafx.scene.canvas.Canvas;
import javafx.scene.canvas.GraphicsContext;
import javafx.scene.control.Button;
import javafx.scene.control.Label;
import javafx.scene.image.WritableImage;
import javafx.scene.paint.Color;


import org.tensorflow.Graph;
import org.tensorflow.Session;
import org.tensorflow.Tensor;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.awt.image.WritableRaster;
import java.io.File;
import java.io.IOException;
import java.net.URISyntaxException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Date;


public class Controller {


    @FXML
    private Label result;
    @FXML
    private Button draw;
    @FXML
    private Button recognize;
    @FXML
    private Button clear;
    @FXML
    private Canvas drawPad;
    @FXML
    private GraphicsContext context;


    //Method for drawing on canvas
    @FXML
    private void getContext() {
        context = drawPad.getGraphicsContext2D();
/*        Another option: draw white numbers on black canvas
        context.setFill(Color.BLACK);
        context.fillRect(0,0,drawPad.getWidth(),drawPad.getHeight());*/
        context.setStroke(Color.BLACK);
        context.setLineWidth(7);

        drawPad.setOnMousePressed(e -> {
            context.beginPath();
            context.lineTo(e.getX(), e.getY());
            context.stroke();
        });

        drawPad.setOnMouseDragged(e -> {
            context.lineTo(e.getX(), e.getY());
            context.stroke();
        });
    }

    //Method for clearing the canvas and result label
    @FXML
    private void clearPad() throws IOException {
        result.setText("");
        if (context == null) {
            return;
        } else {
            context.clearRect(0, 0, drawPad.getWidth(), drawPad.getHeight());
        }

    }

    //Method to convert the drawings on canvas to a 28*28 grayscale png image
    private BufferedImage getImage() throws IOException {
        WritableImage drawn = new WritableImage((int) drawPad.getWidth(), (int) drawPad.getHeight());
        WritableImage snapshot = drawPad.snapshot(new SnapshotParameters(), drawn);
        BufferedImage second = SwingFXUtils.fromFXImage(snapshot, null);
        BufferedImage finalImage = new BufferedImage(28, 28, BufferedImage.TYPE_BYTE_GRAY);
        finalImage.getGraphics().drawImage(second.getScaledInstance(28, 28, Image.SCALE_SMOOTH), 0, 0, null);
        //optional: for viewing converted images, write the png files to a folder
/*        File imageFile = new File(Paths.get("").toAbsolutePath().toString() + "/src/convertedImage/" + new Date().getTime() + ".png");
        ImageIO.write(finalImage, "png", imageFile);*/
        return finalImage;
    }

    //Method to get an array from converted image as input of the trained model
    private float[][][][] getArray(BufferedImage img) {
        float[][][][] array = new float[1][28][28][1];
        WritableRaster raster = img.getRaster();
        DataBufferByte data = (DataBufferByte) raster.getDataBuffer();
        byte[] pixelsData = data.getData();
        for (int i = 0; i < 28; i++) {
            for (int j = 0; j < 28; j++) {
                /*the number is between 0-1 so divided by 255;
                need to get write number on black canvas instead of black number on white canvas, so use 1.0f-*/
                array[0][i][j][0] = 1 - ((float) (pixelsData[i * 28 + j] & 0xFF)) / 255.0f;
            }
        }
        return array;
    }

    /* Method to predict number based on the array
       use pb file (model) saved from Pycharm using tensorflow*/
    private int preditNumber(float[][][][] array) throws IOException {
        Path modelpath = Paths.get(Paths.get("").toAbsolutePath().toString() + "/src/model/mnist.pb");
        byte[] modelBytes = Files.readAllBytes(modelpath);
        try (Graph graph = new Graph()) {
            graph.importGraphDef(modelBytes);
            //Open session to using imported graph/model
            try (Session session = new Session(graph)) {
                Tensor inputData = Tensor.create(array, Float.class);
                float[][] output = new float[1][10];
                Tensor outputData = session.runner().feed("x", inputData).fetch("output").run().get(0);
                outputData.copyTo(output);
                //Get the recognized number (index) with the largest possibility
                float temp = -0.1f;
                int number = 11;
                for (int i = 0; i < 10; i++) {
                    if (output[0][i] > temp) {
                        temp = output[0][i];
                        number = i;
                    }
                }
                //If the possibility is two low, just mark the result as "?", unknown, e.g. a blank canvas
                if (output[0][number] < 0.2) {
                    return -1;
                }
                return number;
            }
        }

    }

    //Final method for recognizing numbers, which involves three methods
    @FXML
    private void recognizeImage() throws IOException {
        if (context == null) {
            result.setText("?");
        } else {
            BufferedImage finalImage = getImage();
            float[][][][] array = getArray(finalImage);
            int numberRecognized = preditNumber(array);
            if (numberRecognized == -1) {
                result.setText("?");
            } else {
                result.setText(String.valueOf(numberRecognized));
            }
        }
    }
}
