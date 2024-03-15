import os
from icecream import ic
import urllib.request
from bs4 import BeautifulSoup
import markdownify


class CustomMarkdownConverter(markdownify.MarkdownConverter):
    def convert_a(self, el, text, convert_as_inline):
        classList = el.get("class")
        if classList and "searched_found" in classList:
            # custom transformation
            # unwrap child nodes of <a class="searched_found">
            text = ""
            for child in el.children:
                text += super().process_tag(child, convert_as_inline)
            return text
        # default transformation
        return super().convert_a(el, text, convert_as_inline)


def load_webpage(link, temp_file: str = r'temp.html', overwrite=False) -> str:

    if os.path.exists(temp_file) and not overwrite:
        with open(temp_file, 'rb') as f:
            content = f.read()
        return content.decode('utf-8', "replace")
    page = urllib.request.urlopen(link)
    content = page.read()
    with open(temp_file, 'wb') as f:
        f.write(content)
    return content.decode('utf-8', "replace")


def scrape_audi_manual():
    # get site map
    root = r'https://www.audia4b9.com'
    link = r'{}/sitemap.html'.format(root)
    main_html = load_webpage(link)
    main_soup = BeautifulSoup(main_html, 'html.parser')
    page_filter = ['{}.html'.format(1816 + i) for i in range(698)]
    count = 1

    all_content = str()
    for item in main_soup.find_all('a'):
        if 'href' not in item.attrs:
            continue

        href_page = item.attrs['href'][item.attrs['href'].rfind('-')+1:]
        if href_page not in page_filter:
            continue
        page_name = item.attrs['href']
        url = '{}{}'.format(root, page_name)
        print('processing {:4}/{:4}: {}'.format(count, len(page_filter), page_name))
        html = load_webpage(url, f'tempdata/{page_name}')
        md = CustomMarkdownConverter().convert(html)
        all_content += '\n\n{}\n'.format(url)
        all_content += md
        count += 1
        if count == 20:
            break
    all_content = all_content.replace('\n\n', '\n').replace('\n\n', '\n')
    # save entire content in text file
    with open('audia4b.md', 'w') as f:
        f.write(all_content)


if __name__ == '__main__':
    scrape_audi_manual()
