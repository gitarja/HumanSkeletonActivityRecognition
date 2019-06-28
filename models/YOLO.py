import cv2
class YOLO:

    def __init__(self, weights, config):
        self.dnn = cv2.dnn.readNet(weights, config)
        self.dnn.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.dnn.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    def get_output_layers(self, net):
        layer_names = self.net.getLayerNames()

        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

        return output_layers

    def forward(self, input):
        blob = cv2.dnn.blobFromImage(input, 1 / 255.0, (416, 416),
                                     swapRB=True, crop=False)
        self.dnn.setInput(blob)
        outs = self.dnn.forward()

        return outs


