#!/usr/bin/env python3
"""Validate generated documentation links and images without a browser/network."""
from html.parser import HTMLParser
import json
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
    manifest = json.loads((SITE / 'site-manifest.json').read_text())
    files = manifest['files']
    errors, count = [], 0
    pages = {}
    for entry in files:
        path = (SITE / entry).resolve()
        if not path.is_relative_to(SITE.resolve()) or not path.is_file():
            errors.append(f'Manifest: invalid or missing file {entry}')
        elif path.suffix == '.html':
            pages[path] = Page(path)
    if not pages:
        errors.append('No documentation pages were built')
    for path, page in pages.items():
        label = path.relative_to(SITE.resolve())
        if len(page.ids) != len(set(page.ids)):
            errors.append(f'{label}: duplicate section IDs')
        for img in page.images:
            if not img.get('alt', '').strip():
                errors.append(f'{label}: missing image description: {img.get("src")}')
        for link in page.links:
            u = urlsplit(link)
            if u.scheme or u.netloc:
                continue
            target = (path.parent / unquote(u.path)).resolve() if u.path else path
            count += 1
            if not target.is_relative_to(SITE.resolve()):
                errors.append(f'{label}: link leaves portable site: {link}')
            elif not target.is_file():
                errors.append(f'{label}: missing {link}')
            elif target.suffix == '.html' and u.fragment:
                destination = pages.get(target)
                if destination is None:
                    errors.append(f'{label}: link to an unbuilt page: {link}')
                elif unquote(u.fragment) not in destination.ids:
                    errors.append(f'{label}: missing anchor {link}')
    if errors:
        raise SystemExit('\n'.join(errors))
    print(f'Documentation: {len(pages)} pages, {count} local links checked; images and anchors valid.')


if __name__ == '__main__':
    main()
