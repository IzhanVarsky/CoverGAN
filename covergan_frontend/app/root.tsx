import type { MetaFunction } from "@remix-run/node";
import {
  Links,
  LiveReload,
  Meta,
  Outlet,
  Scripts,
  ScrollRestoration,
} from "@remix-run/react";
import { useState } from 'react';

export const meta: MetaFunction = () => ({
  charset: "utf-8",
  title: "CoverGAN service",
  viewport: "width=device-width,initial-scale=1",
});

export default function App() {
  const [selectedCover, setSelectedCover] = useState(0);
  // const [covers, setCovers] = useState([
  //   {
  //     src: "https://cdn.imgbin.com/8/12/13/imgbin-phonograph-record-lp-record-album-compact-disc-music-vinyl-YvPihK713U2yvZwgXWRYRYKEW.jpg",
  //     svg: `<svg xmlns="http://www.w3.org/2000/svg" width="512" height="512" viewBox="0 0 512 512"><style>@import url(&#x27;https://fonts.googleapis.com/css?family=PT Serif&#x27;);</style><rect width="512" height="512" fill="rgb(31, 71, 97)" fill-opacity="1"/><path d="M -96,321 C 489,440 90,124 143,304 C -29,84 488,118 356,307 C 338,159 198,320 331,337 C 160,414 -242,222 -241,456 Z" fill="rgb(226, 195, 82)" fill-opacity="0.9450980392156862" stroke="rgb(252, 207, 250)" stroke-opacity="0.9372549019607843" stroke-width="0"/><path d="M -152,523 C -55,309 38,284 363,73 C 149,446 433,408 158,-64 C 253,379 -112,187 547,437 C 593,354 323,-45 229,194 Z" fill="rgb(78, 134, 157)" fill-opacity="0.054901960784313725" stroke="rgb(255, 255, 254)" stroke-opacity="0.00392156862745098" stroke-width="3"/><path d="M -82,-133 C 61,217 -45,449 202,243 C 328,201 584,116 272,194 C 155,460 274,-30 159,388 C 306,494 283,548 -7,-53 Z" fill="rgb(77, 34, 58)" fill-opacity="0.20784313725490197" stroke="rgb(255, 255, 255)" stroke-opacity="0.0196078431372549" stroke-width="4"/><text x="117" y="499" font-family="PT Serif" font-weight="700" font-stretch="normal" font-size="50" textLength="276" font-style="italic" writing-mode="lr" fill="rgb(224, 184, 158)" fill-opacity="1">XXX – YYY</text><circle r="2" cx="117" cy="499" fill="green"/><circle r="2" cx="117" cy="509" fill="lightgreen"/><circle r="2" cx="117" cy="457" fill="yellow"/><circle r="2" cx="117" cy="472" fill="black"/><circle r="2" cx="117" cy="509" fill="rgb(255, 0, 0)"/><circle r="2" cx="117" cy="524" fill="pink"/></svg>`
  //   }
  // ]);
  const [covers, setCovers] = useState([]);

  return (
    <html lang="en">
    <head>
      <script src="https://ajax.googleapis.com/ajax/libs/jquery/3.6.0/jquery.min.js"></script>
      <Meta/>
      <Links/>
    </head>
    <body>
    <Outlet context={[selectedCover, setSelectedCover, covers, setCovers]}/>
    <ScrollRestoration/>
    <Scripts/>
    <LiveReload/>
    </body>
    </html>
  );
}
