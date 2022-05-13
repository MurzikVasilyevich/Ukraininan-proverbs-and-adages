import concurrent.futures
import math
import os
from uuid import uuid4
import shutil

import cv2
import pandas as pd
import pytesseract
import requests
from pdf2image import convert_from_bytes

import settings
from text_tools import text_cleaner, text_filter


class PdfFile:
    def __init__(self, file_url, pdf_folder, first_page, last_page, morph_rect, threshold):
        self.pdf_folder = pdf_folder
        self.folders = {
            'pages_img': create_folder(os.path.join(self.pdf_folder, "pages_img")),
            'pages_OCR': create_folder(os.path.join(self.pdf_folder, "pages_OCR")),
            'quotes_img': create_folder(os.path.join(self.pdf_folder, "quotes_img"))
        }
        self.file_url = file_url
        self.first_page = first_page
        self.last_page = last_page
        self.morph_rect = morph_rect
        self.threshold = threshold
        self.thread_count = (int(math.ceil((last_page - first_page) * 0.1)) if last_page - first_page > 5 else 2) \
            if settings.THREADING else 1
        self.file_name = os.path.basename(file_url)
        self.pages = []
        self.results = []
        self.get_pages()
        shutil.rmtree(self.folders["pages_img"], ignore_errors=True)

    def get_pages(self):
        print(f"Downloading {self.file_url}")
        pdf = requests.get(self.file_url, stream=True)
        print(f"Working in {self.thread_count} threads") if settings.THREADING else print("Working in 1 thread")
        pages = convert_from_bytes(pdf.raw.read(),
                                   first_page=self.first_page, last_page=self.last_page,
                                   dpi=300, thread_count=self.thread_count, fmt="png")
        print(f"Processing {len(pages)} pages")
        if settings.THREADING:
            with concurrent.futures.ThreadPoolExecutor() as page_executor:
                page_executor.map(self.process_page_file, pages, range(len(pages)))
        else:
            for i, page in enumerate(pages):
                self.process_page_file(page, i)

    def process_page_file(self, image, i):
        filepath_rel = os.path.join(self.folders["pages_img"], str(i).zfill(3) + '.png')
        image.save(filepath_rel, 'png')
        print(f'Saving page {i} to {filepath_rel}')
        if settings.OCR:
            page = Page(self, filepath_rel, i)
            self.pages.append(page)
        os.remove(filepath_rel)


class Page:
    def __init__(self, pdf, file_path, page_number):
        self.pdf = pdf
        self.page_number = page_number
        self.file_path = file_path
        self.image = cv2.imread(self.file_path)
        self.contours = self.get_contours()
        self.results = []
        self.get_results()

    def get_contours(self):
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (7, 7), 0)
        thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, self.pdf.morph_rect)
        dilate = cv2.dilate(thresh, kernel, iterations=4)

        contours = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.contours = contours[0] if len(contours) == 2 else contours[1]
        return self.contours

    def get_results(self):
        if settings.THREADING:
            with concurrent.futures.ThreadPoolExecutor() as contour_executor:
                contour_executor.map(self.get_text, self.contours, range(len(self.contours)))
        else:
            for i, contour in enumerate(self.contours):
                self.get_text(contour, i)
        df = pd.DataFrame(self.results)
        df.sort_values(by=['page', 'contour'], ascending=[True, True], inplace=True)
        df_text = df['text']
        df_text.to_csv(os.path.join(self.pdf.folders["pages_OCR"], str(self.page_number).zfill(3) + '.csv'))

    def get_text(self, contour, contour_id):
        image_height, image_width, channels = self.image.shape
        x, y, w, h = cv2.boundingRect(contour)
        if w > self.pdf.threshold[0] and h > self.pdf.threshold[1]:
            region_id = str(self.page_number).zfill(3) + "_" + str(contour_id).zfill(2)
            cropped_image = self.image[y:y + h, x:x + w]
            cropped_file_path = os.path.join(self.pdf.folders["quotes_img"],
                                             region_id + '.png')
            print(f'Saving contour {contour_id} to {cropped_file_path}')
            text = pytesseract.image_to_string(cropped_image, lang=settings.OCR_LANG, config='--psm 1')
            text = text_cleaner(text)
            if not text_filter(text):
                return False
            cv2.imwrite(cropped_file_path, cropped_image)
            print(text)
            bbox = Bbox(x, y, w, h, image_width, image_height)
            result = {
                'region_id': region_id,
                'text': text,
                'page': self.page_number,
                'contour': contour_id,
                'x': bbox.x,
                'y': bbox.y,
                'width': bbox.width,
                'height': bbox.height
            }
            self.results.append(result)
            self.pdf.results.append(result)


class Bbox:
    def __init__(self, x, y, w, h, image_width, image_height, rotation=0):
        self.x = 100 * x / image_width
        self.y = 100 * y / image_height
        self.width = 100 * w / image_width
        self.height = 100 * h / image_height
        self.rotation = rotation


class PdfFiles:
    def __init__(self, sources_file):
        self.sources_file = sources_file
        self.files = []
        self.get_files()

    def get_files(self):
        create_folder(folder=settings.DATA_FOLDER)
        sources = pd.read_csv(self.sources_file)
        for index, row in sources.iterrows():
            file_url = row['file_url']
            pdf_folder = os.path.join(settings.DATA_FOLDER, row['alias'])
            alias = row['alias']
            first_page = int(row['first_page'])
            last_page = int(row['last_page'])
            morph_rect = eval(row['MORPH_RECT'])
            threshold = eval(row['THRESHOLD'])
            os.makedirs(pdf_folder, exist_ok=True)
            pdf_file = PdfFile(file_url, pdf_folder, first_page, last_page, morph_rect, threshold)
            self.files.append(pdf_file)
            if settings.OCR:
                df = pd.DataFrame(pdf_file.results)
                df.sort_values(by=['page', 'contour'], ascending=[True, True], inplace=True)
                df.to_csv(os.path.join(settings.DATA_FOLDER, f'{alias}.csv'), index=False)


def main():
    PdfFiles(settings.SOURCES_FILE)


def create_folder(folder):
    os.makedirs(folder) if not os.path.exists(folder) else None
    return folder


if __name__ == '__main__':
    main()
