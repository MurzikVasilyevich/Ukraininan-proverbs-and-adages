import concurrent.futures
import math
import os
from uuid import uuid4

import cv2
import pandas as pd
import pytesseract
import requests
from pdf2image import convert_from_bytes

import settings


class PdfFile:
    def __init__(self, file_url, pages_folder, first_page, last_page):
        self.file_url = file_url
        self.first_page = first_page
        self.last_page = last_page
        self.thread_count = 5 if last_page - first_page > 5 else int(math.ceil((last_page - first_page) * 0.1))
        self.pages_folder = pages_folder
        self.file_name = os.path.basename(file_url)
        self.pages = []
        self.results = []
        self.get_pages()

    def get_pages(self):
        pdf = requests.get(self.file_url, stream=True)
        pages = convert_from_bytes(pdf.raw.read(),
                                   first_page=self.first_page, last_page=self.last_page,
                                   dpi=300, thread_count=self.thread_count, fmt="png")

        with concurrent.futures.ThreadPoolExecutor() as page_executor:
            page_executor.map(self.save_page_file, pages, range(len(pages)))

    def save_page_file(self, image, i):
        filepath_rel = os.path.join(self.pages_folder,
                                    str(i).zfill(3) + '.png')
        image.save(filepath_rel, 'png')
        page = Page(self, filepath_rel, i)
        # print(f'Saving page {page.page_number} to {filepath_rel}')
        self.pages.append(page)


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

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (12, 10))
        dilate = cv2.dilate(thresh, kernel, iterations=4)

        contours = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.contours = contours[0] if len(contours) == 2 else contours[1]
        return self.contours

    def get_results(self):
        with concurrent.futures.ThreadPoolExecutor() as contour_executor:
            contour_executor.map(self.get_text, self.contours, range(len(self.contours)))

    def get_text(self, contour, contour_id):
        image_height, image_width, channels = self.image.shape
        x, y, w, h = cv2.boundingRect(contour)
        if w > 100 and h > 20:
            cropped_image = self.image[y:y + h, x:x + w]
            text = pytesseract.image_to_string(cropped_image, lang='ukr', config='--psm 1').replace('\n', ' ')
            print(text)
            bbox = {
                'x': 100 * x / image_width,
                'y': 100 * y / image_height,
                'width': 100 * w / image_width,
                'height': 100 * h / image_height,
                'rotation': 0
            }
            region_id = str(uuid4())
            result = {
                'region_id': region_id,
                'text': text,
                'page': self.page_number,
                'contour': contour_id,
                'x': bbox['x'],
                'y': bbox['y'],
                'width': bbox['width'],
                'height': bbox['height']
            }
            self.results.append(result)
            self.pdf.results.append(result)


class PdfFiles:
    def __init__(self, sources_file):
        self.sources_file = sources_file
        self.files = []
        self.get_files()

    def get_files(self):
        sources = pd.read_csv(self.sources_file)
        for index, row in sources.iterrows():
            file_url = row['file_url']
            pages_folder = row['alias']
            alias = row['alias']
            first_page = int(row['first_page'])
            last_page = int(row['last_page'])
            os.makedirs(pages_folder, exist_ok=True)
            pdf_file = PdfFile(file_url, pages_folder, first_page, last_page)
            self.files.append(pdf_file)
            df = pd.DataFrame(pdf_file.results)
            # df.sort_values(by=['page', 'contour'], ascending=[True, False], inplace=True)
            df.to_csv(os.path.join(settings.RESULTS_FOLDER, f'{alias}.csv'), index=False)


def main():
    os.makedirs(settings.RESULTS_FOLDER, exist_ok=False)
    PdfFiles(settings.SOURCES_FILE)


if __name__ == '__main__':
    main()
