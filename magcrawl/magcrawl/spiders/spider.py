# -*- coding: utf-8 -*-
# install deltafetch https://www.lfd.uci.edu/~gohlke/pythonlibs/#bsddb3
import scrapy
from w3lib.html import remove_tags

class WebSpider(scrapy.Spider):
    name = 'spider'
    # user_agent = 'Mozilla/5.0 (Windows NT 6.3; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/44.0.2403.157 Safari/537.36'
    # download_delay = 5.0
    allowed_domains = ['-']
    start_urls = ['-']


    def parse_article(self, response):
        title = response.xpath("//h1[@class='evo-entry-title evo-sticky-title']//text()").extract()
        url = response.xpath("//html//head//meta[8]//@content").extract()
        published_time = response.xpath("//html/head/meta[@property='article:published_time']//@content").extract()
        description = response.xpath("//html/head/meta[@property='og:description']/@content").extract()
        category = response.xpath("//div[@class='evo-post-header evo-single-bottom']//div[@class='evo-post-meta top']//div[@class='evo-post-cat']//a//text()").extract()
        contentxpath = response.xpath("//article[@itemtype='http://schema.org/Article']//div[@class='evo-entry-content entry-content evo-dropcrap']")
        tagsxpath = response.xpath("//div[@class='evo-tags']")
        contentlist= []
        for p in contentxpath.xpath('.//p'):
            c = p.get().replace("<p>", "").replace("</p>", "").lstrip()
            contentlist.append(c)
        tagslist = []
        for t in tagsxpath.xpath('.//li//a//text()').extract():
            tagslist.append(t)


        contentstring = ' '.join(contentlist)
        content = remove_tags(contentstring)
        tagsstring = ','.join(tagslist)
        tags = remove_tags(tagsstring)

        scraped_info = {
                'title' : title,
                'url' : url,
                'published_time' : published_time,
                'description' : description,
                'category': category,
                'content': content,
                'tags': tags
            }
        yield scraped_info


    def parse(self, response):
        
        next_url = response.xpath("/html/body/div[1]/div[1]/div/div[3]/div[1]/div/div/div/div[2]/div[3]/nav/div/a[5]//@href")

        for article_url in response.xpath('/html/body/div[1]/div[1]/div/div[3]/div[1]/div/div/div/div[2]/div[2]/article//h3//a//@href').extract():
            # print("ENTERING URL------" + article_url)
            yield response.follow(article_url, callback=self.parse_article)

        next_page = response.xpath(".//div[@class='nav-links']//a[@class='next page-numbers']//@href").extract_first()
        if next_page is not None:
        	print("GOING TO NEXT PAGE:" + next_page)
        	yield response.follow(next_page, callback=self.parse)



    



