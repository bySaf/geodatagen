#!/usr/bin/env python3
"""Check the portable documentation entry points without a browser or network."""
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / 'user_review'


class Page(HTMLParser):
    def __init__(self, path):
        super().__init__()
        self.links, self.ids, self.images = [], [], []
        self.feed(path.read_text(encoding='utf-8'))

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if 'id' in attrs:
            self.ids.append(attrs['id'])
        for name in ('src', 'href'):
            if attrs.get(name):
                self.links.append(attrs[name])
        if tag == 'img':
            self.images.append(attrs)


def main():
    errors, count = [], 0
    for path in (SITE/'index.html', SITE/'realism/index.html', SITE/'realism/guide.html'):
        page = Page(path)
        if len(page.ids) != len(set(page.ids)):
            errors.append(f'{path.name}: duplicate section IDs')
        for img in page.images:
            if not img.get('alt', '').strip():
                errors.append(f'{path.name}: missing image description: {img.get("src")}')
        for link in page.links:
            u = urlsplit(link)
            if u.scheme or u.netloc:
                continue
            target = (path.parent / unquote(u.path)).resolve() if u.path else path
            count += 1
            if not target.is_relative_to(SITE):
                errors.append(f'{path.name}: link leaves portable site: {link}')
            elif not target.is_file():
                errors.append(f'{path.name}: missing {link}')
            elif target.suffix == '.html' and u.fragment and unquote(u.fragment) not in Page(target).ids:
                errors.append(f'{path.name}: missing anchor {link}')
    if errors:
        raise SystemExit('\n'.join(errors))
    print(f'Documentation: {count} local links checked; image descriptions and section IDs valid.')


if __name__ == '__main__':
    main()
