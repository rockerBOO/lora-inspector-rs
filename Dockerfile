FROM pierrezemb/gostatic
COPY index.html /srv/http/index.html
COPY main.css /srv/http/main.css
COPY main.js /srv/http/main.js
COPY pkg /srv/http/pkg
COPY favicon-32.png /srv/http/favicon-32.png
COPY favicon-16.png /srv/http/favicon-16.png
COPY favicon-32.ico /srv/http/favicon-32.ico
COPY apple-touch-icon.png /srv/http/apple-touch-icon.png
COPY logo.png /srv/http/logo.png
COPY chartist-plugin-pointlabels.min.js /srv/http/chartist-plugin-pointlabels.min.js
CMD ["-port","8080","-https-promote", "-enable-logging"]
