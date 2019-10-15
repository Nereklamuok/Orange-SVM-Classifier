import modbus_tk.defines as cst
import modbus_tk
import sklearn as skl
import tkinter as tk
import numpy as np
import serial
import time
import glob
import cv2
import os

from serial.tools import list_ports
from modbus_tk import modbus_rtu
from PIL import Image, ImageTk
from tkinter import filedialog
from sklearn.externals import joblib
from sklearn import svm
from skimage import feature

#Inicialização de variáveis globais
#Parâmetros LBP
LBP_raio = 2
LBP_pontos = 8 * LBP_raio

np.set_printoptions(threshold=np.inf)

class Sample(object):
    def __init__(self, _img, _label):
        self.img = _img
        self.label = _label
        self.feature_vector = []
        self.other_features = []

    def process_sample(self):
        #--------------------------------------------Processamento da imagem--------------------------------------------
        #-Normalização da imagem para o intervalo 0 a 255-
        img = cv2.normalize(self.img, None, alpha=0, beta=255, norm_type = cv2.NORM_MINMAX)
        #-Redimensionamento da imagem para 256x256 pelo método de interpolação de área-
        img = cv2.resize(img, (256, 256), interpolation = cv2.INTER_AREA)
        #-Remoção de ruído da imagem-
        blur_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        blur_img = cv2.GaussianBlur(blur_img, (7, 7), 0)
        #-Divisão dos canais da imagem-
        _,_,b = cv2.split(blur_img)

        #-Segmentação da imagem através de limiarização do canal B (método de Otsu)-
        th, mask = cv2.threshold(b, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        cv2.imwrite('maskb.jpeg', mask)

        #-Gera um kernel elipsóide 3x3-
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))    
        #-Aplica 5 iterações da operação morfológica de fechamento na máscara, utilizando o kernel criado-
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations = 5) 

        #-Obtem os contornos da máscara-
        _, cnt, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        #-Encontra o maior contorno em relação à sua área-
        max_cnt = max(cnt, key = cv2.contourArea)
        #-Define os parâmetros do menor retângulo que englobe o maior contorno encontrado-
        x, y, w, h = cv2.boundingRect(max_cnt)

        #-Combinação das imagens com a máscara para obtenção da Região de Interesse (ROI)-
        #-Imagem original (BGR)-
        img = cv2.bitwise_and(img, img, mask = mask)

        #-Recorta as imagens utilizando os parâmetros do retângulo obtido anteriormente-
        #-Recorte da imagem original-
        img = img[y:y+h, x:x+w]
        self.img_masked = img
        #-Recorta também a máscara utilizando os mesmos parâmetros-
        mask_rsz = mask[y:y+h, x:x+w]
        #-Redimensionamento das imagens para 256x256 pelo método de interpolação cúbica-
        #-Redimensionamento da imagem original-        
        img = cv2.resize(img, (256, 256), interpolation = cv2.INTER_CUBIC)
        #-Redimensionamento da imagem L*a*b*-
        LAB_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        #-Redimensionamento da máscara-
        mask_rsz = cv2.resize(mask_rsz, (256, 256), interpolation = cv2.INTER_NEAREST)

        #----------------------------------------Fim do processamento da imagem-----------------------------------------

        #------------------------------------------Extração de características------------------------------------------
        b, g, r = cv2.split(img)
        LAB_L, LAB_a, LAB_b = cv2.split(LAB_img)
        mask_rsz = cv2.bitwise_not(mask_rsz)
        #-------------DESCRITOR DE COR------------        
        LAB_a_ma = np.ma.masked_array(LAB_a, mask=mask_rsz)
        LAB_b_ma = np.ma.masked_array(LAB_b, mask=mask_rsz)

        LAB_a_hist = np.histogram(LAB_a_ma.compressed().ravel(), bins= np.arange(0, 256), density = True)
        LAB_b_hist = np.histogram(LAB_b_ma.compressed().ravel(), bins= np.arange(0, 256), density = True)
        #-----------DESCRITOR DE TEXTURA----------
        #-Extrair o descritor de textura (Local Binary Patterns) dos canais da imagem RGB-
        LBP_b = feature.local_binary_pattern(b, LBP_pontos, LBP_raio, 'uniform')
        LBP_g = feature.local_binary_pattern(g, LBP_pontos, LBP_raio, 'uniform')
        LBP_r = feature.local_binary_pattern(r, LBP_pontos, LBP_raio, 'uniform')

        LBP_b_ma = np.ma.masked_array(LBP_b, mask=mask_rsz)
        LBP_g_ma = np.ma.masked_array(LBP_g, mask=mask_rsz)
        LBP_r_ma = np.ma.masked_array(LBP_r, mask=mask_rsz)

        #-Gerar o histograma do LBP dos canais RGB-
        LBP_hist_b = np.histogram(LBP_b_ma.compressed().ravel(), bins= np.arange(0, LBP_b_ma.max() + 1), density = True)
        LBP_hist_g = np.histogram(LBP_g_ma.compressed().ravel(), bins= np.arange(0, LBP_b_ma.max() + 1), density = True)
        LBP_hist_r = np.histogram(LBP_r_ma.compressed().ravel(), bins= np.arange(0, LBP_b_ma.max() + 1), density = True)

        #-----------DESCRITOR DE TAMANHO----------
        #-Obtém o elipsoide que melhor se encaixa ao contorno-
        try:
            elipse = cv2.fitEllipseAMS(max_cnt)
        except cv2.error as e:
            print("ERRO ao obter elipse")
        else:
        #Extrai os valores dos eixos da elipse
            _, self.other_features, _ = elipse
        #----------VETOR DE CARACTERÍSTICAS---------
        #-Expande o vetor de características com o histograma dos canais 'a' e 'b' da imagem L*a*b-
        self.feature_vector.extend(LAB_a_hist[0].flatten())
        self.feature_vector.extend(LAB_b_hist[0].flatten())
        #-Expande o vetor de características com o histograma da imagem LBP-
        self.feature_vector.extend(LBP_hist_b[0].flatten())
        self.feature_vector.extend(LBP_hist_g[0].flatten())
        self.feature_vector.extend(LBP_hist_r[0].flatten())


class ClassifierWindow(tk.Toplevel):
    def __init__(self, parent):
        tk.Toplevel.__init__(self, parent)
        self.parent = parent

        self.window_width = 260
        self.window_height = 240

        self.pos_x = (self.parent.screen_width/2) - (self.window_width/2)
        self.pos_y = (self.parent.screen_height/2) - (self.window_height/2)

        self.geometry('%dx%d+%d+%d' % (self.window_width, self.window_height, self.pos_x, self.pos_y))
        self.resizable(False, False)
        self.title("Treinar")

        self.dir_list = []
        self.class_list = []
        self.opt_menu_str = tk.StringVar()

        self.label1 = tk.Label(self, text = "Selecione diretórios de amostras:")
        self.label1.place(x = 0, y = 15, width = 180, height = 20)

        self.button_dir = tk.Button(self, text="Procurar", command = self.search_dir)
        self.button_dir.place(x = 190, y = 10, width = 60, height = 30)

        self.label2 = tk.Label(self, text = "Classes selecionadas:")
        self.label2.place(x = 0, y = 50, width = 120, height = 20)

        self.opt_menu = tk.OptionMenu(self, self.opt_menu_str, [])
        self.opt_menu.place(x = 128, y = 45, width = 125, height = 30)

        self.button_remove = tk.Button(self, text="Remover", command = self.remove_entry)
        self.button_remove.place(x = 130, y = 80, width = 120, height = 25)

        self.button_train = tk.Button(self, text="Treinar classificador", command = self.start_learning)
        self.button_train.place(x = 60, y = 115, width = 120, height = 25)

        self.button_load = tk.Button(self, text="Carregar classificador", command = self.load_classifier)
        self.button_load.place(x = 60, y = 145, width = 120, height = 25)

        self.button_save = tk.Button(self, text="Salvar classificador", command = self.save_classifier)
        self.button_save.place(x = 60, y = 175, width = 120, height = 25)
        
        self.button_return = tk.Button(self, text="Voltar", command = self.destroy)
        self.button_return.place(x = 60, y = 205, width = 120, height = 25)

        self.focus_get()
        self.grab_set()

    def search_dir(self):
        self.dirname =  filedialog.askdirectory(title = "Select the directory where the class' samples are located")
        if self.dirname:
            self.dir_list.append("".join([self.dirname,"/*.jpeg"]))
            self.dirname = os.path.basename(os.path.normpath(self.dirname))
            self.class_list.append(self.dirname)
            self.refresh_optionmenu()

    def refresh_optionmenu(self):
        menu = self.opt_menu["menu"]
        menu.delete(0, "end")

        for string in self.class_list:
            menu.add_command(label=string, command = lambda value = string: self.opt_menu_str.set(value))

        if self.class_list:
            self.opt_menu_str.set(self.class_list[0])
        else:
            self.opt_menu_str.set("")

    def remove_entry(self):
        menu = self.opt_menu["menu"]
        index_num = menu.index(self.opt_menu_str.get())
        del self.class_list[index_num]
        del self.dir_list[index_num]
        self.refresh_optionmenu()

    def start_learning(self):
        if len(self.dir_list) <= 1:
            self.parent.features = []
            self.parent.labels = []
            PopupWindow(self, "ERRO", "Número de classes ou características insuficiente!\r\n"
                "Certifique-se de que os vetores de características não sejam vazios e "
                "de que existam ao menos duas classes.", 200, 180)
            return
        self.parent.features = []
        self.parent.labels = []

        for index, dir in enumerate(self.dir_list):
            for file in glob.glob(dir):
                print("OK")
                print(file)
                sample_ = Sample(cv2.imread(file), index)
                sample_.process_sample()
                self.parent.features.append(sample_.feature_vector)
                self.parent.labels.append(sample_.label)
                print('OK2')

        try:
            self.parent.classifier.fit(self.parent.features, self.parent.labels)
            print(self.parent.classifier.predict(self.parent.features))
        except ValueError:
            self.parent.features = []
            self.parent.labels = []
            PopupWindow(self, "ERRO", "Número de classes ou características insuficiente!\r\n"
                "Certifique-se de que os vetores de características não sejam vazios e "
                "de que existam ao menos duas classes.", 200, 180)
        else:
            PopupWindow(self, "Treinamento", "Treinamento concluído!", 100, 100)

    def load_classifier(self):
        load_file = filedialog.askopenfilename(title = "Select file", filetypes = (("Pikle files", "*pkl"),))
        if load_file:
            self.parent.classifier = joblib.load(load_file)
            print(self.parent.classifier.classes_)
            PopupWindow(self, "Classificador", "Classificador carregado!", 100, 100)
        else:
            print("ERRO")

    def save_classifier(self):
        try:
            skl.utils.validation.check_is_fitted(self.parent.classifier, 'support_')
        except skl.exceptions.NotFittedError:
            popup = PopupWindow(self, "ERRO", "O classificador ainda não foi treinado!", 100, 100)   
        else:
            save_file = filedialog.asksaveasfilename(title = "Save as...", filetypes = (("Pikle files", "*pkl"),), defaultextension = ".pkl")
            if save_file:
                joblib.dump(self.parent.classifier, save_file)
                popup = PopupWindow(self, "Salvando classificador", "Classificador salvo com sucesso!", 100, 100)   
            else:
                print("Cancelado")

class VideoCapture():
    def __init__(self, parent, video_source=0):
        self.parent = parent
        self.video_cap = cv2.VideoCapture(video_source)
        if not self.video_cap.isOpened():
            raise ValueError("Não é possível acessar o dispositivo de reprodução.", video_source)

        self.video_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.video_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        self.width = 640
        self.height = 480

    def __del__(self):
        if self.video_cap.isOpened():
            self.video_cap.release()
        self.parent.mainloop()

    def get_frame(self, RGB):
        if self.video_cap.isOpened():
            _, frame = self.video_cap.read()

            if frame.any():
                frame = cv2.flip(frame, 1)
                if RGB:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return(frame)

class ConnectionWindow(tk.Toplevel):
    def __init__(self, parent):
        tk.Toplevel.__init__(self, parent)
        self.parent = parent

        self.port_list = []
        self.selected_port = tk.StringVar()

        self.protocol('WM_DELETE_WINDOW', self.close_window) 

        self.geometry('%dx%d' % (260, 330))
        self.resizable(False, False)

        self.title("Conexão")

        self.config_frame = tk.LabelFrame(self, text = 'Configuração  de comunicação')
        self.config_frame.place(x = 10, y = 10, width = 240, height = 310)

        self.label1 = tk.Label(self.config_frame, text = "Porta de comunicação:")
        self.label1.place(x = 15, y = 15, width = 130)

        self.ports_menu = tk.OptionMenu(self.config_frame, self.selected_port, [])
        self.ports_menu.place(x = 145, y = 10, width = 80)
        self.refresh_ports()

        self.label2 = tk.Label(self.config_frame, text = "Baud rate:")
        self.label2.place(x = 85, y = 46, width = 60)

        self.selected_baudrate = tk.IntVar(value = 57600)
        self.baudrate_menu = tk.OptionMenu(self.config_frame, self.selected_baudrate, 1200, 2400, 4800, 9600, 19200, 28800, 38400, 57600)
        self.baudrate_menu.place(x = 145, y = 41, width = 80)

        self.label3 = tk.Label(self.config_frame, text = "Paridade:")
        self.label3.place(x = 65, y = 77, width = 60)

        self.selected_parity = tk.StringVar(value = "Nenhuma")
        self.parity_menu = tk.OptionMenu(self.config_frame, self.selected_parity, "Nenhuma", "Par", "Impar")
        self.parity_menu.place(x = 125, y = 72, width = 100)

        self.label4 = tk.Label(self.config_frame, text = "Data bits:")
        self.label4.place(x = 105, y = 108, width = 60)

        self.selected_databits = tk.IntVar(value = 8)
        self.databits_menu = tk.OptionMenu(self.config_frame, self.selected_databits, 7, 8)
        self.databits_menu.place(x = 165, y = 103, width = 60)

        self.label5 = tk.Label(self.config_frame, text = "Stop bits:")
        self.label5.place(x = 105, y = 139, width = 60)

        self.selected_stopbits = tk.IntVar(value = 1)
        self.stopbits_menu = tk.OptionMenu(self.config_frame, self.selected_stopbits, 1, 2)
        self.stopbits_menu.place(x = 165, y = 134, width = 60)

        self.label6 = tk.Label(self.config_frame, text = "Timeout (ms):")
        self.label6.place(x = 85, y = 170)

        self.selected_timeout = tk.StringVar()
        self.selected_timeout.set(1000)
        vcmd = self.register(self.validate_input)
        self.timeout_textbox = tk.Entry(self.config_frame, textvariable = self.selected_timeout, validate = 'all', validatecommand = (vcmd, '%d','%S', '%P'))
        self.timeout_textbox.place(x = 170, y = 170, width = 50, height = 20)

        self.label7 = tk.Label(self.config_frame, text = "max 10s", fg = "gray50", font = ("TkDefaultFont", 7))
        self.label7.place(x = 170, y = 195, height = 10)

        self.refresh_button = tk.Button(self.config_frame, text = "Atualizar", command = self.refresh_ports)
        self.refresh_button.place(x = 85, y = 200, width = 80)

        if self.parent.serial_connected == False:
            self.connect_button = tk.Button(self.config_frame, text = "Conectar", command = self.connect_port)
            self.connect_button.place(x = 85, y = 230, width = 80)
        else:
            self.connect_button = tk.Button(self.config_frame, text = "Desconectar", command = self.disconnect_port)
            self.connect_button.place(x = 85, y = 230, width = 80)

        self.return_button = tk.Button(self.config_frame, text = "Voltar", command = self.destroy)
        self.return_button.place(x = 85, y = 260, width = 80)

        self.focus_get()
        self.grab_set()

    def validate_input(self, input_type, value, new_value):
        if int(input_type) == 1:
            if value.isdigit():
                if int(new_value) > 10000 or int(new_value) <= 0:
                    return False
                else:
                    return True
            else:
                return False
        else:
            return True

    def refresh_ports(self):
        self.port_list = []
        for ports in list_ports.comports():
            self.port_list.append(ports.device)

        menu = self.ports_menu["menu"]
        menu.delete(0, "end")

        for string in self.port_list:
            menu.add_command(label=string, command = lambda value = string: self.selected_port.set(value))

        if self.port_list:
            self.selected_port.set(self.port_list[0])
        else:
            self.selected_port.set("")

    def connect_port(self):
        try:
            self.parent.master_rtu = modbus_rtu.RtuMaster(serial.Serial(port = self.selected_port.get(), baudrate = self.selected_baudrate.get(), bytesize = self.selected_databits.get(), parity = 'N', stopbits = self.selected_stopbits.get()))
            
            if self.parent.master_rtu._is_opened:
                self.parent.master_rtu.close()
            
            self.parent.master_rtu.set_timeout(1)
            self.parent.master_rtu.set_verbose(True)
            self.parent.master_rtu.open()

            self.parent.master_rtu.execute(1, cst.WRITE_SINGLE_COIL, 7000, output_value = 1)

        except Exception as ex:
            template = "An exception of type {0} occurred. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)

            PopupWindow(self, "Conexão", message, 300, 150)

        else:
            self.parent.serial_connected = True
            self.parent.etapa_processamento = 0
            self.connect_button.configure(text = "Desconectar", command = self.disconnect_port)

            PopupWindow(self, "Conexão", "Dispositivo conectado!", 100, 100)


    def disconnect_port(self):
        try:
            self.parent.master_rtu.execute(1, cst.WRITE_SINGLE_COIL, 7000, output_value = 0)

            self.parent.master_rtu.close()
            self.parent.master_rtu._do_close()

        except Exception as ex:
            template = "An exception of type {0} occurred. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)

            PopupWindow(self, "Conexão", message, 300, 150)
        else:
            self.parent.serial_connected = False
            self.parent.etapa_processamento = 0
            self.connect_button.configure(text = "Conectar", command = self.connect_port)

            PopupWindow(self, "Conexão", "Dispositivo desconectado!", 100, 100)

    def close_window(self):
        self.destroy()


class InspectWindow(tk.Toplevel):
    def __init__(self, parent):
        tk.Toplevel.__init__(self, parent)
        self.parent = parent
        self.cam = VideoCapture(self, 0)
        self.serial_connected = False

        self.port_list = []
        self.selected_port = tk.StringVar()
        self.etapa_processamento = 0

        self.geometry('%dx%d' % (950, 550))
        self.resizable(False, False)

        self.title("Inspeção")
        self.protocol('WM_DELETE_WINDOW', self.close_window)     
        
        self.video_frame = tk.LabelFrame(self, text = "Câmera", width = self.cam.width, height = self.cam.height)
        self.video_frame.place(x = 20, y = 20)

        self.lmain = tk.Label(self.video_frame)
        self.lmain.grid()

        self.modbus_frame = tk.LabelFrame(self, text = "Modbus", width = 240, height = 150)
        self.modbus_frame.place(x = self.cam.width + 40, y = 20)

        self.button_connect = tk.Button(self.modbus_frame, text="Conexão", command = self.open_ConnectionWindow)
        self.button_connect.place(x = 70, y = 0, width = 80, height = 30)
        
        self.label_modbus = tk.Label(self.modbus_frame, text = "MODBUS: OFF")
        self.label_modbus.place(x = 70, y = 35, width = 100)

        self.label_pist1 = tk.Label(self.modbus_frame, text = "PISTAO 1: OFF", anchor = tk.W)
        self.label_pist1.place(x = 10, y = 50, width = 100)

        self.label_pist2 = tk.Label(self.modbus_frame, text = "PISTAO 2: OFF", anchor = tk.W)
        self.label_pist2.place(x = 10, y = 65, width = 100)

        self.label_pist3 = tk.Label(self.modbus_frame, text = "PISTAO 3: OFF", anchor = tk.W)
        self.label_pist3.place(x = 10, y = 80, width = 100)

        self.label_esteira = tk.Label(self.modbus_frame, text = "ESTEIRA: OFF", anchor = tk.W)
        self.label_esteira.place(x = 110, y = 50, width = 100)

        self.label_ilum = tk.Label(self.modbus_frame, text = "ILUMINAÇÃO: OFF", anchor = tk.W)
        self.label_ilum.place(x = 110, y = 65, width = 120)

        self.sample_frame = tk.LabelFrame(self, text = "Resultado", width = 240, height = 350)
        self.sample_frame.place(x = self.cam.width + 40, y = 173)

        self.sample_img = tk.Label(self.sample_frame)
        self.sample_img.place(x = 10, y = 10, width = 220, height = 220)

        self.label_class = tk.Label(self.sample_frame, text = "Classe: ", anchor = tk.W)
        self.label_class.place(x = 10, y = 240)

        self.button_analyze = tk.Button(self.sample_frame, text = "Analisar amostra", command = self.capture_sample)
        self.button_analyze.place(x = 60, y = 290 , width = 120, height = 30)

        self.update_camera()
        self.update_serial()

        self.focus_get()
        self.grab_set()
        
    def open_ConnectionWindow(parent):
        ConnectionWindow(parent)

    def update_camera(self):
        frame = self.cam.get_frame(True)

        if frame.any():
            frame = cv2.resize(frame, (self.cam.width, self.cam.height), interpolation = cv2.INTER_AREA)
            img = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image = img)
            self.lmain.imgtk = imgtk
            self.lmain.configure(image=imgtk)         

        self.after(10, self.update_camera)

    def capture_sample(self):
        result = -1
        frame = self.cam.get_frame(False)
        sample_ = Sample(frame, 0)
        sample_.process_sample()

        img_rsz = cv2.resize(sample_.img_masked, (220, 220), interpolation = cv2.INTER_CUBIC)
        img_rsz = cv2.cvtColor(img_rsz, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img_rsz)
        imgtk = ImageTk.PhotoImage(image = img)
        self.sample_img.place(x = 10, y = 10, width = 220, height = 220)
        self.sample_img.imgtk = imgtk
        self.sample_img.configure(image = imgtk)

        ftrs = []
        ftrs.append(sample_.feature_vector)
        try:
            self.result = self.parent.classifier.predict(ftrs)
        except skl.exceptions.NotFittedError:
            PopupWindow(self, "ERRO", "O classificador ainda não foi treinado!", 100, 100)
        else:
            self.label_class.configure(text = "Classe: "+str(self.result))
            return self.result

    def update_serial(self):
        amostra_presente = 0
        resultado_analise = -1

        if self.serial_connected == True:
            if self.etapa_processamento == 0:
                inicializado = 0
                try:
                    inicializado = self.master_rtu.execute(1, cst.READ_COILS, 7000, 1)[0]
                except Exception as ex:
                    template = "An exception of type {0} occurred. Arguments:\n{1!r}"
                    message = template.format(type(ex).__name__, ex.args)
                    print(message)
                else:
                    if inicializado == 1:
                        self.etapa_processamento = 1
            if self.etapa_processamento == 1:
                try:
                    #print(self.master_rtu.execute(1, cst.DIAGNOSTIC, 2))
                    amostra_presente = self.master_rtu.execute(1, cst.READ_DISCRETE_INPUTS, 1, 1)[0]
                except Exception as ex:
                    template = "An exception of type {0} occurred. Arguments:\n{1!r}"
                    message = template.format(type(ex).__name__, ex.args)
                    print(message)
                else:
                    if amostra_presente == 1:
                        #time.sleep(1)
                        resultado_analise = self.capture_sample()
                        if resultado_analise != -1:
                            for x in range(0, 10):
                                try:
                                    self.master_rtu.execute(1, cst.WRITE_SINGLE_COIL, 7001, output_value = 1)
                                except Exception as ex:
                                    template = "An exception of type {0} occurred. Arguments:\n{1!r}"
                                    message = template.format(type(ex).__name__, ex.args)
                                    print(message)
                                    pass
                                else:
                                    print(resultado_analise)
                                    for x in range(0, 10):
                                        try:
                                            if(resultado_analise == self.parent.classifier.classes_[0]):
                                                self.master_rtu.execute(1, cst.WRITE_SINGLE_COIL, 7002, output_value = 1)
                                            elif(resultado_analise == self.parent.classifier.classes_[1]):
                                                self.master_rtu.execute(1, cst.WRITE_SINGLE_COIL, 7003, output_value = 1)
                                            elif(resultado_analise == self.parent.classifier.classes_[2]):
                                                self.master_rtu.execute(1, cst.WRITE_SINGLE_COIL, 7004, output_value = 1)
                                            else:
                                                self.master_rtu.execute(1, cst.WRITE_SINGLE_COIL, 7005, output_value = 1)
                                        except Exception as ex:
                                            template = "An exception of type {0} occurred. Arguments:\n{1!r}"
                                            message = template.format(type(ex).__name__, ex.args)
                                            print(message)
                                            pass
                                        else:
                                            self.etapa_processamento = 2
                                            break
                                    break
            if self.etapa_processamento == 2:
                try:
                    amostra_presente = self.master_rtu.execute(1, cst.READ_DISCRETE_INPUTS, 1, 1)[0]
                except Exception as ex:
                    template = "An exception of type {0} occurred. Arguments:\n{1!r}"
                    message = template.format(type(ex).__name__, ex.args)
                    print(message)
                else:
                    if amostra_presente == 0:
                        self.etapa_processamento = 1;

            try:
                self.inputs = self.master_rtu.execute(1, cst.READ_DISCRETE_INPUTS, 0, 6)
                self.outputs = self.master_rtu.execute(1, cst.READ_COILS, 3000, 6)
            except Exception as ex:
                    template = "An exception of type {0} occurred. Arguments:\n{1!r}"
                    message = template.format(type(ex).__name__, ex.args)
                    print(message)
            else:
                if self.inputs[0] == 1:
                    self.label_modbus.config(text = "MODBUS: ON")
                else:
                    self.label_modbus.config(text = "MODBUS: OFF")

                if self.outputs[0] == 1:
                    self.label_pist1.config(text = "PISTAO 1: ON")
                else:
                    self.label_pist1.config(text = "PISTAO 1: OFF")

                if self.outputs[1] == 1:
                    self.label_pist2.config(text = "PISTAO 2: ON")
                else:
                    self.label_pist2.config(text = "PISTAO 2: OFF")

                if self.outputs[2] == 1:
                    self.label_pist3.config(text = "PISTAO 3: ON")
                else:
                    self.label_pist3.config(text = "PISTAO 3: OFF")

                if self.outputs[4] == 1:
                    self.label_esteira.config(text = "ESTEIRA: ON")
                else:
                    self.label_esteira.config(text = "ESTEIRA: OFF")

                if self.outputs[5] == 1:
                    self.label_ilum.config(text = "ILUMINAÇÃO: ON")
                else:
                    self.label_ilum.config(text = "ILUMINAÇÃO: OFF")

        self.after(100, self.update_serial)

    def close_window(self):
        self.cam.video_cap.release()
        if hasattr(self, 'master_rtu'):
            try:
                self.master_rtu.execute(1, cst.WRITE_SINGLE_COIL, 7000, output_value = 0)

                self.master_rtu.close()
                self.master_rtu._do_close()

            except Exception as ex:
                template = "An exception of type {0} occurred. Arguments:\n{1!r}"
                message = template.format(type(ex).__name__, ex.args)
                print(message)
            else:
                print("Conexões fechadas")
        self.destroy()

class AboutWindow(tk.Toplevel):
    def __init__(self, parent):
        tk.Toplevel.__init__(self, parent)
        self.parent = parent

        self.title("Sobre")
        self.protocol('WM_DELETE_WINDOW', self.destroy) 

        self.screen_width = 250
        self.screen_height = 300

        self.label = tk.Label(self, text = "NADA")
        self.label.place(x = 80, y = 80)

class PopupWindow(tk.Toplevel):
    def __init__(self, parent, _title, _text, _width, _height):
        tk.Toplevel.__init__(self, parent)
        self.parent = parent

        self.screen_width = self.winfo_screenwidth()
        self.screen_height = self.winfo_screenheight()

        self.window_width = _width
        self.window_height = _height

        self.pos_x = (self.screen_width/2) - (self.window_width/2)
        self.pos_y = (self.screen_height/2) - (self.window_height/2)

        self.geometry('%dx%d+%d+%d' % (self.window_width, self.window_height, self.pos_x, self.pos_y))
        self.title(_title)

        self.msg = tk.Message(self, text = _text, justify = tk.CENTER, width = self.window_width-(self.window_width/5))
        self.msg.pack(pady = self.window_height/20)

        self.button = tk.Button(self, text = "Ok", command = self.close_popup)
        self.button.pack()

        self.bind('<Return>', lambda _: self.close_popup())

        self.focus_set()
        self.grab_set()

    def close_popup(self):
        self.parent.focus_set()
        self.parent.grab_set()
        self.destroy()

class MainWindow(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)

        self.labels = []
        self.features = []
        self.classifier = svm.SVC(kernel = 'rbf', C=20, gamma = 0.8)

        self.title("Inspetor de qualidade de citrus v0.1")
        self.iconbitmap(default = "icone_laranja.ico")
        self.protocol('WM_DELETE_WINDOW', self.close_window) 

        self.screen_width = self.winfo_screenwidth()
        self.screen_height = self.winfo_screenheight()

        self.window_width = 130
        self.window_height = 210

        self.pos_x = (self.screen_width/2) - (self.window_width/2)
        self.pos_y = (self.screen_height/2) - (self.window_height/2)

        self.geometry('%dx%d+%d+%d' % (self.window_width, self.window_height, self.pos_x, self.pos_y))

        self.resizable(False, False)

        self.label_title = tk.Label(self, text="VisiOrange", font = ("Fixedsys", 8))
        self.label_title.place(x = 20, y = 5)

        self.label1 = tk.Label(self, text="Selecione uma opção:", anchor = tk.W, justify = tk.CENTER)
        self.label1.place(x = 4, y = 30)

        self.button_classifier = tk.Button(self, text="Classificador", command = self.open_ClassifierWindow)
        self.button_classifier.place(x = 25, y = 60, width = 80, height = 30)

        self.button_inspect = tk.Button(self, text="Inspeção", command = self.open_InspectWindow)
        self.button_inspect.place(x = 25, y = 100, width = 80, height = 30)

        self.button_about = tk.Button(self, text="Sobre", command = self.open_AboutWindow)
        self.button_about.place(x = 25, y = 140, width = 80, height = 30)

        self.label_vers = tk.Label(self, text="v0.1", anchor = tk.SE, justify = tk.RIGHT, fg = "gray50", font = ("Courier", 8))
        self.label_vers.place(x = 98, y = 193)

        #self.update_idletasks()
        #print(self.label2.winfo_width())
        #print(self.label2.winfo_height())

    def open_ClassifierWindow(parent):
        ClassifierWindow(parent)

    def open_InspectWindow(parent):
        try: 
            skl.utils.validation.check_is_fitted(parent.classifier, 'support_')
        except skl.exceptions.NotFittedError:
            PopupWindow(parent, "ERRO", "O classificador ainda não foi treinado!", 100, 100)
            return
        else:
            InspectWindow(parent)

    def open_AboutWindow(parent):
        AboutWindow(parent)

    def close_window(self):
        self.destroy()

if __name__ == "__main__":
    root = MainWindow()
    root.mainloop()