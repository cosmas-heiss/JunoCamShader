import numpy as np
import os
import shutil
from bs4 import BeautifulSoup
from urllib.request import Request, urlopen
import requests
from zipfile import ZipFile
from PIL import Image
import json
import time
import datetime
from datetime import timedelta
from datetime import datetime as dt


# this is the threshold above which the image is deemed usable
MEAN_IMAGE_VALUE_THRESHOLD = 10.0


def is_relevant_dates(a, b, all_image_times):
    link_start= dt.strptime(a, '%y%m%d') - timedelta(days=1)
    link_end = dt.strptime(b, '%y%m%d') + timedelta(days=1)
    for start_time, end_time in all_image_times:
        if not (link_start >= end_time or link_end <= start_time):
            return True
    return False



def unzip_tmp_files(tmp_path, save_to_path):
    with ZipFile(tmp_path, 'r') as zip_file:
        for content in zip_file.namelist():
            zip_file.extractall(save_to_path)


def check_raw_image(raw_image_paths):
    if len(raw_image_paths) != 1:
        return False
    img = Image.open(raw_image_paths[0])
    if not (img.size[0] == 1648 and img.size[1] % 384 == 0):
        return False
    return True


class PerijoveDataDownloader:
    def __init__(self, gui_object=None):
        self.MEAN_IMAGE_VALUE_THRESHOLD = 1.0
        self.data_path = os.path.join(os.getcwd(), 'Data')
        self.keep_doing_download = True

        self.gui_object = gui_object


    def set_gui_object(self, gui_object):
        self.gui_object = gui_object


    def get_perijove_range(self):
        image_download_path = "https://www.missionjuno.swri.edu/junocam/processing?source=junocam&ob_from=&ob_to=&phases[]=PERIJOVE+1&perpage=1"
        try:
            req = Request(image_download_path)
            html_page = urlopen(req)
        except Exception as e:
            return None

        soup = BeautifulSoup(html_page, "lxml")
        perijove_links = [x.text for x in soup.findAll('label') if 'Perijove' in x.text]
        available_perijoves = [int(x.split()[1]) for x in perijove_links]
        return (min(available_perijoves), max(available_perijoves))


    def download_image_and_data(self, image_href, data_href, tmp_path):
        response_image = self.do_request("https://www.missionjuno.swri.edu" + image_href, prog_bar_number=0)
        if not self.keep_doing_download: return None, None
        image_path_tmp = os.path.join(tmp_path, 'tmp_image.zip')
        with open(image_path_tmp, "wb") as save_file:
            save_file.write(response_image.content)
        
        response_data = self.do_request("https://www.missionjuno.swri.edu" + data_href, prog_bar_number=0)
        if not self.keep_doing_download: return None, None
        data_path_tmp  = os.path.join(tmp_path, 'tmp_data.zip')
        with open(data_path_tmp, "wb") as save_file:
            save_file.write(response_data.content)

        return image_path_tmp, data_path_tmp


    def do_request(self, url, prog_bar_number=2):
        counter = 0
        while counter <= 5:
            if not self.keep_doing_download: return None
            try:
                data = requests.get(url)
                return data
            except Exception as e:
                counter += 1
                if counter > 5:
                    self.keep_doing_download = False
                    self.cleanup_temp_folder()
                    self.gui_object.close_download_progress_bar()
                    self.gui_object.show_could_not_download_error()
                self.gui_object.update_download_progress_bar(prog_bar_number, 0.0, 'Request failed, retry {}/{}'.format(counter, 5))
                time.sleep(1)


    def mkdir_if_not_exist(self, path):
        if not os.path.exists(path):
            os.mkdir(path)


    def remove_if_exists(self, path):
        if os.path.exists(path):
            os.remove(path)


    def cleanup_temp_folder(self):
        tmp_path = os.path.join(self.data_path, 'tmp')

        self.remove_if_exists(os.path.join(tmp_path, 'tmp_data.zip'))
        self.remove_if_exists(os.path.join(tmp_path, 'tmp_image.zip'))

        for el in self.downloaded_tmp_stuff:
            self.remove_if_exists(el)

        if os.path.exists(os.path.join(tmp_path, 'images')):
            if len(os.listdir(os.path.join(tmp_path, 'images'))) == 0:
                os.rmdir(os.path.join(tmp_path, 'images'))

        if os.path.exists(os.path.join(tmp_path, 'images_info')):
            if len(os.listdir(os.path.join(tmp_path, 'images_info'))) == 0:
                os.rmdir(os.path.join(tmp_path, 'images_info'))

        if os.path.exists(tmp_path):
            if len(os.listdir(tmp_path)) == 0:
                os.rmdir(tmp_path)

    def exit_download(self):
        self.keep_doing_download = False

    def download(self, perijove_number):
        error_code = 0
        self.keep_doing_download = True
        self.downloaded_tmp_stuff = []
        self.gui_object.start_download_progress_bar(perijove_number, exit_command=self.exit_download)
        self.gui_object.update_download_progress_bar(0, 0.0, text="Downloading images: ...")
        self.gui_object.update_download_progress_bar(1, 0.0, text="Selecting usable images: ...")
        self.gui_object.update_download_progress_bar(2, 0.0, text="Downloading SPICE kernels: ...")

        assert os.path.exists(self.data_path)
        tmp_path = os.path.join(self.data_path, 'tmp')

        global_kernel_path = os.path.join(self.data_path, 'global_kernels')
        if not os.path.exists(global_kernel_path):
            os.mkdir(global_kernel_path)


        self.mkdir_if_not_exist(tmp_path)
        self.mkdir_if_not_exist(os.path.join(tmp_path, 'images'))
        self.mkdir_if_not_exist(os.path.join(tmp_path, 'images_info'))

        perijove_path = os.path.join(self.data_path, 'PJ{:02d}'.format(perijove_number))

        if not os.path.exists(perijove_path):
            os.mkdir(perijove_path)

        self.mkdir_if_not_exist(perijove_path)
        self.mkdir_if_not_exist(os.path.join(perijove_path, 'images'))
        self.mkdir_if_not_exist(os.path.join(perijove_path, 'images_info'))
        self.mkdir_if_not_exist(os.path.join(perijove_path, 'spice_kernels'))

        ### DONWLOADING PART ###
        image_download_path = "https://www.missionjuno.swri.edu/junocam/processing?source=junocam&ob_from=&ob_to=&phases[]=PERIJOVE+{}&perpage=100".format(perijove_number)
        req = Request(image_download_path)
        html_page = urlopen(req)

        soup = BeautifulSoup(html_page, "lxml")
        links = []
        for link in soup.findAll('a'):
            href = link.get('href')
            if len(href.split('id=')) == 2 and "/junocam/processing?id=" in href:
                links.append(href)

        if not self.keep_doing_download: return 0
        self.gui_object.update_download_progress_bar(0, 100.0 / (len(links) + 1) - 1e-5)
        for k, link in enumerate(links):
            if not self.keep_doing_download: return 0
            self.gui_object.update_download_progress_bar(0, 0.0, text="Downloading images: {}".format("https://www.missionjuno.swri.edu" + link))
            req = Request("https://www.missionjuno.swri.edu" + link)
            html_page = urlopen(req)
            soup = BeautifulSoup(html_page, "lxml")

            sublink_image, sublink_metadata = None, None
            for sublink in soup.findAll('a'):
                href = sublink.get('href')
                if "/Vault/VaultDownload?" in href:
                    if 'Metadata' in sublink.text.strip() and sublink_metadata is None:
                        sublink_metadata = href
                    elif 'Images' in sublink.text.strip() and sublink_image is None:
                        sublink_image = href

            if sublink_image is None or sublink_metadata is None:
                continue

            save_path_image, save_path_data = self.download_image_and_data(sublink_image, sublink_metadata, tmp_path)
            if not self.keep_doing_download: return 0

            with ZipFile(save_path_image, 'r') as zip_file:
                for member in zip_file.namelist():
                    img_name = member.split('/')[-1]
                    if img_name[-4:] == '.png' and 'raw' in img_name:
                        source = zip_file.open(member)
                        target = open(os.path.join(tmp_path, 'images', img_name), "wb")
                        self.downloaded_tmp_stuff.append(os.path.join(tmp_path, 'images', img_name))
                        with source, target:
                            shutil.copyfileobj(source, target)

            with ZipFile(save_path_data, 'r') as zip_file:
                for member in zip_file.namelist():
                    json_name = member.split('/')[-1]
                    if json_name[-5:] == '.json':
                        source = zip_file.open(member)
                        target = open(os.path.join(tmp_path, 'images_info', json_name), "wb")
                        self.downloaded_tmp_stuff.append(os.path.join(tmp_path, 'images_info', json_name))
                        with source, target:
                            shutil.copyfileobj(source, target)

            if not self.keep_doing_download: return 0
            self.gui_object.update_download_progress_bar(0, 100.0 / (len(links) + 1) - 1e-5)

        if not self.keep_doing_download: return 0
        self.gui_object.update_download_progress_bar(0, 0.0, text="Downloading images: DONE")
        ### END DOWNLOADING PART ###


        ### SELECTION PART ###
        all_image_times = []

        info_files = [os.path.join(tmp_path, 'images_info', x) for x in os.listdir(os.path.join(tmp_path, 'images_info'))]
        for info_file in info_files:
            if not self.keep_doing_download: return 0
            self.gui_object.update_download_progress_bar(1, 100.0 / len(info_files) - 1e-5)

            with open(info_file, 'r') as file_content:
                json_dict = json.load(file_content)

            if json_dict['FILTER_NAME'] != ['BLUE', 'GREEN', 'RED']:
                continue
            if json_dict['LINE_SAMPLES'] != 1648:
                continue
            if json_dict['LINES'] % 384 != 0:
                continue

            name = json_dict['PRODUCT_ID']
            image_name = json_dict['FILE_NAME']

            img = Image.open(os.path.join(tmp_path, 'images', image_name))
            if np.mean(img) < MEAN_IMAGE_VALUE_THRESHOLD:
                continue

            if not self.keep_doing_download: return 0
            self.gui_object.update_download_progress_bar(1, 0.0, text="Selecting usable images: " + image_name)

            start_time = json_dict['START_TIME']
            stop_time = json_dict['STOP_TIME']
            all_image_times.append((dt.strptime(start_time + '000', '%Y-%m-%dT%H:%M:%S.%f'), dt.strptime(stop_time + '000', '%Y-%m-%dT%H:%M:%S.%f')))

            shutil.move(info_file, os.path.join(perijove_path, 'images_info', name + '.json'))
            shutil.move(os.path.join(tmp_path, 'images', image_name), os.path.join(perijove_path, 'images', name + '.png'))
        if not self.keep_doing_download: return 0
        self.gui_object.update_download_progress_bar(1, 0.0, text="Selecting usable images: DONE")
        ### END SELECTION PART ###


        ### DOWNLOADING KERNELS PART ###
        if not self.keep_doing_download: return 0
        self.gui_object.update_download_progress_bar(2, 10.0 - 1e-5, text="Downloading SPICE kernels: /spk")


        data_download_path = "https://naif.jpl.nasa.gov/pub/naif/pds/data/jno-j_e_ss-spice-6-v1.0/jnosp_1000/data/spk/"
        req = Request(data_download_path)
        html_page = urlopen(req)

        soup = BeautifulSoup(html_page, "lxml")
        links = []
        for link in soup.findAll('a'):
            href = link.get('href')
            link_text = link.text.strip()
            if link_text[-4:] == '.bsp' and link_text[:9] == 'juno_rec_' and is_relevant_dates(link_text.split('_')[2], link_text.split('_')[3], all_image_times):
                links.append(href)

        if len(links) == 0:
            error_code = 1

        for link in links:
            response_data = self.do_request(data_download_path + link)
            if not self.keep_doing_download: return 0
            kernel_path = os.path.join(perijove_path, 'spice_kernels', link)
            with open(kernel_path, "wb") as save_file:
                save_file.write(response_data.content)
        if not self.keep_doing_download: return 0
        self.gui_object.update_download_progress_bar(2, 10.0 - 1e-5)

        if not self.keep_doing_download: return 0
        self.gui_object.update_download_progress_bar(2, 0.0, text="Downloading SPICE kernels: /ck")
        data_download_path = "https://naif.jpl.nasa.gov/pub/naif/pds/data/jno-j_e_ss-spice-6-v1.0/jnosp_1000/data/ck/"
        req = Request(data_download_path)
        html_page = urlopen(req)

        soup = BeautifulSoup(html_page, "lxml")
        links = []
        for link in soup.findAll('a'):
            href = link.get('href')
            link_text = link.text.strip()
            if link_text[-3:] == '.bc' and link_text[:12] == 'juno_sc_rec_' and is_relevant_dates(link_text.split('_')[3], link_text.split('_')[4], all_image_times):
                links.append(href)

        if len(links) == 0:
            error_code = 1

        for link in links:
            response_data = self.do_request(data_download_path + link)
            if not self.keep_doing_download: return 0
            kernel_path = os.path.join(perijove_path, 'spice_kernels', link)
            with open(kernel_path, "wb") as save_file:
                save_file.write(response_data.content)
        if not self.keep_doing_download: return 0
        self.gui_object.update_download_progress_bar(2, 10.0 - 1e-5)

        if not self.keep_doing_download: return 0
        self.gui_object.update_download_progress_bar(2, 0.0, text="Downloading SPICE kernels: /sclk")
        data_download_path = "https://naif.jpl.nasa.gov/pub/naif/pds/data/jno-j_e_ss-spice-6-v1.0/jnosp_1000/data/sclk/"
        req = Request(data_download_path)
        html_page = urlopen(req)

        soup = BeautifulSoup(html_page, "lxml")
        links = []
        version_list = []
        for link in soup.findAll('a'):
            href = link.get('href')
            link_text = link.text.strip()
            if link_text[-4:] == '.tsc' and link_text[:13] == 'jno_sclkscet_':
                version_list.append(link_text.split('_')[2].split('.')[0])

        link = 'jno_sclkscet_{}.tsc'.format(max(version_list, key=lambda x:int(x)))
        response_data = self.do_request(data_download_path + link)
        if not self.keep_doing_download: return 0
        kernel_path = os.path.join(perijove_path, 'spice_kernels', link)
        with open(kernel_path, "wb") as save_file:
            save_file.write(response_data.content)
        if not self.keep_doing_download: return 0
        self.gui_object.update_download_progress_bar(2, 10.0 - 1e-5)


        file_links = ["https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/de430.bsp",
                      "https://naif.jpl.nasa.gov/pub/naif/pds/data/jno-j_e_ss-spice-6-v1.0/jnosp_1000/data/fk/juno_v12.tf",
                      "https://naif.jpl.nasa.gov/pub/naif/pds/data/jno-j_e_ss-spice-6-v1.0/jnosp_1000/data/spk/jup310.bsp",
                      "https://naif.jpl.nasa.gov/pub/naif/pds/data/jno-j_e_ss-spice-6-v1.0/jnosp_1000/data/lsk/naif0012.tls",
                      "https://naif.jpl.nasa.gov/pub/naif/pds/data/jno-j_e_ss-spice-6-v1.0/jnosp_1000/data/spk/juno_struct_v04.bsp",
                      "https://naif.jpl.nasa.gov/pub/naif/pds/data/jno-j_e_ss-spice-6-v1.0/jnosp_1000/data/pck/pck00010.tpc"]
        for data_download_path in file_links:
            file_name = data_download_path.split('/')[-1]
            if not self.keep_doing_download: return 0
            self.gui_object.update_download_progress_bar(2, 10.0 - 1e-5, text="Downloading SPICE kernels: " + file_name)
            if os.path.exists(os.path.join(global_kernel_path, file_name)):
                continue
            
            response_data = self.do_request(data_download_path)
            if not self.keep_doing_download: return 0
            kernel_path = os.path.join(global_kernel_path, file_name)
            with open(kernel_path, "wb") as save_file:
                save_file.write(response_data.content)

        ### END DOWNLOADING KERNELS PART ###
        if not self.keep_doing_download: return 0
        self.gui_object.update_download_progress_bar(2, 0.0, text="Deleting temporary files: ...")
        
        self.cleanup_temp_folder()

        if not self.keep_doing_download: return 0
        self.gui_object.update_download_progress_bar(2, 0.0, text="Deleting temporary files: DONE")
        if not self.keep_doing_download: return 0
        self.gui_object.close_download_progress_bar()
        return error_code

