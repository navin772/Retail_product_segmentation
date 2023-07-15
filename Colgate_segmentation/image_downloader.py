from icrawler.builtin import GoogleImageCrawler

google_crawler = GoogleImageCrawler(
    feeder_threads=1,
    parser_threads=2,
    downloader_threads=6,
    storage={'root_dir': 'colgate_shelf'})
google_crawler.crawl(keyword='colgate on shelf', max_num=10)