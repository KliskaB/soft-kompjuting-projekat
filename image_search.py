from google_images_download import google_images_download


class GoogleImagesDownloader:

    def download_image(self, words):
        response = google_images_download.googleimagesdownload()
        arguments = {"keywords": words,
                     "limit": 5,
                     "print_urls": True,
                     'chromedriver':
                         r"C:\Users\Bojana\Desktop\chromedriver\chromedriver.exe"}
        try:
            paths = response.download(arguments)
            print(paths)
        except:
            pass
