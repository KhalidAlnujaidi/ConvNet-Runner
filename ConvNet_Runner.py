from asyncio.windows_events import NULL
import os
import sys
import matplotlib.pylab as plt
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from datetime import datetime
from PyQt5.QtWidgets import *
from CovNet_ui import *
from PyQt5.QtCore import *
from PyQt5.QtGui import QPixmap
tf.get_logger().setLevel('ERROR')

global models

models = {
    "efficientnet_v2": ("https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet1k_b0/feature_vector/2", (224,224)),
    "mobilenet_v3": ("https://tfhub.dev/google/imagenet/mobilenet_v3_large_100_224/feature_vector/5", (224,224)),
    "inception_v3": ("https://tfhub.dev/google/imagenet/inception_v3/feature_vector/5", (299,299)),
    "inception_resnet_v2": ("https://tfhub.dev/google/imagenet/inception_resnet_v2/feature_vector/5", (299,299)),
    "resnet_v2": ("https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/5", (224,224))
    }

BATCH_SIZE = ""
EPOCH = ""
MODEL_CHOICE = ""

INPUT_DIRECTORY = ""
OUTPUT_DIRECTORY = ""



class DlgMain(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(DlgMain, self).__init__(parent=parent)
        self.setupUi(self)
        self.statusBar().showMessage("Developed at Prince Mohamed Bin Fahad University. By Khalid Alnujaidi.")

        # Connect Model Radio Buttons
        self.rdbMB.toggled.connect(self.evnt_MB_checked)
        self.rdbI.toggled.connect(self.evnt_I_checked)
        self.rdbEff.toggled.connect(self.evnt_Eff_checked)
        self.rdbRes.toggled.connect(self.evnt_Res_checked)
        self.rdbResI.toggled.connect(self.evnt_ResI_checked)

        # Connect I/O directory buttons
        self.inputbtn.clicked.connect(self.getInputFolder)
        self.outputbtn.clicked.connect(self.getOutputFolder)

        self.model_get_btn.clicked.connect(self.getModelDir)


       
        # Connect Run button / load img / make prediction
        self.runBTN.clicked.connect(self.getParams)
        self.loadImg_btn.clicked.connect(self.load_img_func)
        self.predict_btn.clicked.connect(self.make_pred_func)

        
        
        


    # get combo box choice
    def get_combx_choice(self):
        global loaded_model_type
        loaded_model_type = self.model_comboBox.currentText()


    # I/O dir innitialization
    def getInputFolder(self):
        global INPUT_DIRECTORY
        INPUT_DIRECTORY = QFileDialog.getExistingDirectory(self, "Image Data Directory")
        self.inputText.setText(INPUT_DIRECTORY)
    def getOutputFolder(self):
        global OUTPUT_DIRECTORY
        OUTPUT_DIRECTORY = QFileDialog.getExistingDirectory(self, "Folder to output the results")
        self.outputText.setText(OUTPUT_DIRECTORY)
        OUTPUT_DIRECTORY = self.outputText.text()

    def getModelDir(self):
        global getModelDir
        getModelDir = QFileDialog.getExistingDirectory(self, "Folder model checkpoint")

        self.model_dir_text.setText(getModelDir)

        
    

    # Functionality of Radio buttons 
    def evnt_MB_checked(self, chk):
        global MODEL_CHOICE
        MODEL_CHOICE = "mobilenet_v3"
            
    def evnt_I_checked(self, chk):
        global MODEL_CHOICE
        MODEL_CHOICE = "inception_v2"
        
    def evnt_Eff_checked(self, chk):
        global MODEL_CHOICE
        MODEL_CHOICE = "efficientnet_v2"
        
    def evnt_Res_checked(self, chk):
        global MODEL_CHOICE
        MODEL_CHOICE = "resnet_v2"
        
    def evnt_ResI_checked(self, chk):
        global MODEL_CHOICE
        MODEL_CHOICE = "inception_resnet_v2"


    def plot_acc(self, list1, list2, dir):
        plt.plot(list1)
        plt.plot(list2)
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['Training accuracy', 'Validation accuracy'], loc='lower right')
        plt.savefig(dir)
        plt.clf()
    def plot_loss(self, list1, list2, dir):
        plt.plot(list1)
        plt.plot(list2)
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['Training loss', 'Validation loss'], loc='upper right')
        plt.savefig(dir)
        plt.clf()
    def plot_mtrx(self,cmtrx,dir):
        fig = plt.figure(figsize = (8,8))
        ax1 = fig.add_subplot(1,1,1)
        sns.set(font_scale=1.4) #for label size
        sns.heatmap(cmtrx, annot=True, annot_kws={"size": 12},
        cbar = False, cmap='Purples')
        ax1.set_ylabel('True Values',fontsize=14)
        ax1.set_xlabel('Predicted Values',fontsize=14)
        plt.savefig(dir)
        plt.clf()
    

    # Load Img Btn
    def load_img_func(self):
        global THE_IMG
        THE_IMG = QFileDialog.getOpenFileName(self, "Open Image")
        print(THE_IMG[0])
        THE_IMG = THE_IMG[0]
        pixmap = QPixmap(THE_IMG)
        self.img_label.setPixmap(pixmap)

    def update_pred_label(self,string_pred):
        self.prediction_label.setText(string_pred)





    
    # make prediction btn
    def make_pred_func(self):



        self.worker = MakePredictions()
        self.worker.start()
        self.worker.send_prediction_label.connect(self.update_pred_label)
        

    # Run button functionality
    def getParams(self):
        global EPOCHS, BATCH_SIZE
        EPOCHS =  self.epochs_spnbx.cleanText() # here
        EPOCHS = int(EPOCHS)
        BATCH_SIZE =  self.batch_size_spnbx.cleanText()

        self.runBTN.setDisabled(True)
        
        # run application
        self.worker = WorkerThread()
        self.worker.start()
        self.worker.send_to_plot_acc.connect(self.plot_acc)
        self.worker.send_to_plot_loss.connect(self.plot_loss)
        self.worker.send_cmrtx.connect(self.plot_mtrx)
        
        
class MakePredictions(QThread):
    send_prediction_label = pyqtSignal(str)
    def run(self):
        global output_prediction

        output_prediction = "..."
        self.send_prediction_label.emit(output_prediction)
        # IMG loaded properlly | checkpoint dir loaded properlly | classes & imgSize loaded properly
        print(os.listdir(getModelDir))

        model = tf.keras.models.load_model(getModelDir) # get trained model
        # get class names
        class_name = []
        f = open(getModelDir + '/class_names.txt', 'r')
        for line in f:
            x = line[:-1]
            class_name.append(x)
        f.close()
        print(class_name)
        # get model img size
        s = open(getModelDir + '/image_size.txt', 'r')
        number = s.readline()
        number = int(number)
        IMG_SIZE = (number, number)
        s.close()
        print(IMG_SIZE)

        img = tf.keras.utils.load_img(THE_IMG, target_size=IMG_SIZE)



        img_array = tf.keras.utils.img_to_array(img)
        img_array = img_array / 255
        img_array = tf.expand_dims(img_array, 0)  # Create a batch

        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])


        output_prediction = "This image most likely belongs to {} with a {:.2f} percent confidence.".format(class_name[np.argmax(score)], 100 * np.max(score))
        #
        # self.prediction_label.setText(output_prediction)
        self.send_prediction_label.emit(output_prediction)

        print(output_prediction)

        # print(
        #     "This image most likely belongs to {} with a {:.2f} percent confidence."
        #     .format(class_name[np.argmax(score)], 100 * np.max(score))
        # )



 ############################## THREAD ################################################################
class WorkerThread(QThread):
     # list / list / str
    send_to_plot_acc = pyqtSignal(list,list,str)
    send_to_plot_loss  = pyqtSignal(list,list,str)
    send_cmrtx = pyqtSignal(np.ndarray, str)
      
    def run(self):
        # get model specs
        feature_extractor, IMAGE_SIZE = models[MODEL_CHOICE]
        print(f" model url --> {feature_extractor}" + f"\nmodel image size ..> {IMAGE_SIZE}")

        # setup data pipelines
        train_dir = INPUT_DIRECTORY + "/train"
        train_ds = tf.keras.preprocessing.image_dataset_from_directory(
            train_dir,
            label_mode="categorical",
            seed=123,
            image_size=IMAGE_SIZE,
            batch_size=1)

        # get name of classes

        global name_of_classes

        name_of_classes = train_ds.class_names
        print(name_of_classes)
        class_names = tuple(train_ds.class_names)
        train_size = train_ds.cardinality().numpy()
        train_ds = train_ds.unbatch().batch(int(BATCH_SIZE))
        train_ds = train_ds.repeat()

        val_dir = INPUT_DIRECTORY + "/val"
        val_ds = tf.keras.preprocessing.image_dataset_from_directory(
            val_dir,
            label_mode="categorical",
            seed=123,
            image_size=IMAGE_SIZE,
            batch_size=1)
        val_size = val_ds.cardinality().numpy()
        val_ds = val_ds.unbatch().batch(int(BATCH_SIZE))

        normalization_layer = tf.keras.layers.Rescaling(1. / 255)
        preprocessing_model = tf.keras.Sequential([normalization_layer])
        preprocessing_model.add(
            tf.keras.layers.RandomRotation(40))
        preprocessing_model.add(
            tf.keras.layers.RandomTranslation(0, 0.2))
        preprocessing_model.add(
            tf.keras.layers.RandomTranslation(0.2, 0))
        preprocessing_model.add(
            tf.keras.layers.RandomZoom(0.2, 0.2))
        preprocessing_model.add(
            tf.keras.layers.RandomFlip(mode="horizontal"))
        preprocessing_model.add(
            tf.keras.layers.RandomFlip(mode="vertical"))
        train_ds = train_ds.map(lambda images, labels:
                            (preprocessing_model(images), labels))
        val_ds = val_ds.map(lambda images, labels:
                            (normalization_layer(images), labels))
        print("train data augmented and normalize")
        print("val data normalized")

        # build and compile model
        model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=IMAGE_SIZE + (3,)),
            hub.KerasLayer(feature_extractor, trainable=True), # Here model is being downloaded
            tf.keras.layers.Dropout(rate=0.2),
            tf.keras.layers.Dense(len(class_names),
                            kernel_regularizer=tf.keras.regularizers.l2(0.0001))
            ])
        model.build((None,)+IMAGE_SIZE+(3,))
        model.summary() 
        model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.9), 
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=0.1),
        metrics=['accuracy'])

        # Saving best model settings
        SaveModelDir= f"{OUTPUT_DIRECTORY}/{MODEL_CHOICE}/exp_for_{str(BATCH_SIZE)}_BatchSize"
        checkpoint_path = SaveModelDir+"/cp.ckpt"
        checkpoint_dir = os.path.dirname(checkpoint_path)





        metric = 'accuracy'
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, monitor=metric, 
                  save_weights_only=False, save_best_only = True, verbose=1)
        model.save(SaveModelDir+f"/model_for_{str(BATCH_SIZE)}_BatchSize")

        # Train model 
        start_time = datetime.now()
        steps_per_epoch = train_size // int(BATCH_SIZE)
        validation_steps = val_size // int(BATCH_SIZE)
        hist = model.fit(
        train_ds,
        epochs=EPOCHS, steps_per_epoch=steps_per_epoch,
        validation_data=val_ds,
        validation_steps=validation_steps,
        callbacks = [cp_callback]
        ).history
        end_time = datetime.now() # end timer
        f = open(SaveModelDir+f"/time_to_train_for_{BATCH_SIZE}_model.txt", "x")
        f.write('Duration: {}'.format(end_time - start_time))
        f.close()




        # save class names ( name_of_classes )
        f = open(checkpoint_path + "/class_names.txt", "x")
        for className in name_of_classes:
            f.write("%s\n" % className)
        f.close()

        # save image size
        f = open(checkpoint_path + "/image_size.txt", "x")

        f.write(str(IMAGE_SIZE[0]))
        f.close()

        # Graph training performance
        acc = hist['accuracy']
        print("^--acc + acc class")
        val_acc = hist['val_accuracy']
        graph_path = SaveModelDir+'/model_acc.png'
        self.send_to_plot_acc.emit(acc,val_acc,graph_path)

        graph_path = SaveModelDir+'/model_loss.png'
        loss = hist['loss']
        val_loss = hist['val_loss']
        graph_path = SaveModelDir+'/model_loss.png'
        self.send_to_plot_loss.emit(loss,val_loss,graph_path)


        # confusion matrix and metracies
        y_pred = []  
        y_true = []
        for image_batch, label_batch in val_ds:   
            y_true.append(label_batch)
            preds = model.predict(image_batch)
            y_pred.append(np.argmax(preds, axis = - 1))
        correct_labels = tf.concat([item for item in y_true], axis = 0)
        predicted_labels = tf.concat([item for item in y_pred], axis = 0)
            # Un-One-hot encode the correct labels
        correct_labels_tonums = np.argmax(correct_labels,axis=1)

        
        

        cm = confusion_matrix(correct_labels_tonums, predicted_labels)
        print(cm)
        print(type(cm))
        save_cm_dir = SaveModelDir+'/con_matrx_val_ds.png'
        self.send_cmrtx.emit(cm, save_cm_dir)

        
        


        loss, acc = model.evaluate(val_ds, verbose=2)
        f = open(SaveModelDir+f"/VAL_DS_{BATCH_SIZE}_model.txt", "x")
        f.write(classification_report(correct_labels_tonums,predicted_labels))
        f.write("\n model accuracy on val_ds {:5.2f}%".format(100 * acc))
        f.close()
        print("Training has been complete :)")
        
        

if __name__ == '__main__':
    app = QApplication(sys.argv)
    dlgMain = DlgMain()
    dlgMain.show()
    sys.exit(app.exec_())