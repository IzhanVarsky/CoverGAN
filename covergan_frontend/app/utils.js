const correct_color_regex =
  "^(#([\\da-f]{3}){1,2}|(rgb|hsl)a\\((\\d{1,3}%?,\\s?){3}(1|1\\.0+|0|0?\\.\\d+)\\)|(rgb|hsl)\\(\\d{1,3}%?(,\\s?\\d{1,3}%?){2}\\))$";
const re = new RegExp(correct_color_regex);

function findColorByAttrName(obj, attr) {
  return obj
    .find(`[${attr}]`)
    .filter((i, obj) => re.test(obj.getAttribute(attr).trim().toLowerCase()))
    .map((i, x) => ({x, attr, value: x.getAttribute(attr)}));
}

function extractColors(parsed) {
  return ["fill", "stroke", "stop-color"]
    .map(name => findColorByAttrName(parsed, name))
    .reduce((flatten, arr) => [...flatten, ...arr]);
}

const getColors = (svg) => {
  return extractColors($(svg)).map(obj => ({
    attr: obj.attr,
    value: obj.value
  }))
}

const changeColorByIndex = (svg, ind, newColor) => {
  const parsed = $(svg);
  const res_objs = extractColors(parsed);
  if (ind < res_objs.length) {
    const obj = res_objs[ind];
    obj.x.setAttribute(obj.attr, newColor);
  }
  return parsed[0].outerHTML;
}

const changeAllColors = (svg, newColors) => {
  const parsed = $(svg);
  const res_objs = extractColors(parsed);
  res_objs.forEach((el, i) => el.x.setAttribute(el.attr, newColors[i]))
  return parsed[0].outerHTML;
}

const getSVGSize = (svg) => {
  try {
    const parsed = $(svg)[0];
    const w = parseInt(parsed.getAttribute("width"));
    const h = parseInt(parsed.getAttribute("height"));
    return {w, h};
  } catch (e) {
    return {w: 0, h: 0}
  }
}

const svgWithSize = (svg, size) => {
  try {
    const parsed = $(svg)[0];
    parsed.setAttribute("width", size);
    parsed.setAttribute("height", size);
    return parsed.outerHTML;
  } catch (e) {
    return svg;
  }
}

function getSVGViewBoxSize(thisSVG) {
  let width;
  let height;
  if (thisSVG.hasAttribute("viewBox")) {
    const viewBox = thisSVG.getAttribute("viewBox").split(" ");
    width = viewBox[2];
    height = viewBox[3];
  } else {
    width = thisSVG.getAttribute("width");
    height = thisSVG.getAttribute("height");
  }
  return {width, height};
}

function addBeforeText(parsed, rect) {
  const find = parsed.find("text:first");
  if (find.length) {
    find.before(rect);
  } else {
    parsed.append(rect);
  }
}

const addShadowFilter = (svg) => {
  const parsed = $(svg);
  let newIdNum = 0;
  let id;
  while (true) {
    id = `id${newIdNum}`;
    if (parsed.find(`[id^=${id}]`).length > 0) {
      newIdNum++;
      continue;
    }
    break;
  }
  const {width, height} = getSVGViewBoxSize(parsed[0]);
  const cx = width / 2;
  const cy = height / 2;
  const r = Math.ceil(width / Math.sqrt(2));
  const radGrad = `
    <radialGradient cx="${cx}" cy="${cy}" r="${r}" 
        gradientUnits="userSpaceOnUse" spreadMethod="pad" id="${id}">
        <stop offset="0" stop-color="rgba(0, 0, 0, 0)"></stop>
        <stop offset="0.7" stop-color="rgba(0, 0, 0, 0)"></stop>
        <stop offset="1" stop-color="rgba(0, 0, 0, 0.6)"></stop>
    </radialGradient>`;
  let findDefs = parsed.find("defs:first");
  if (findDefs.length > 0) {
    findDefs.append(radGrad);
  } else {
    parsed.prepend(`<defs>${radGrad}</defs>`);
  }
  const rect = `<rect x="0" y="0" width="${width}" height="${height}" fill="url(#${id})" />`;
  addBeforeText(parsed, rect);
  return parsed[0].outerHTML;
}
const addRectBefore = (svg, color = 'rgba(230, 230, 230, 0.5)') => {
  const parsed = $(svg);
  let {width, height} = getSVGViewBoxSize(parsed[0]);

  const rect = `<rect x="0" y="0" width="${width}" height="${height}" fill="${color}" />`;
  addBeforeText(parsed, rect);
  return parsed[0].outerHTML;
}

const prettifyXml = function (sourceXml) {
  const xmlDoc = new DOMParser().parseFromString(sourceXml, 'application/xml');
  const xsltDoc = new DOMParser().parseFromString([
    // describes how we want to modify the XML - indent everything
    '<xsl:stylesheet xmlns:xsl="http://www.w3.org/1999/XSL/Transform">',
    '  <xsl:strip-space elements="*"/>',
    '  <xsl:template match="para[content-style][not(text())]">', // change to just text() to strip space in text nodes
    '    <xsl:value-of select="normalize-space(.)"/>',
    '  </xsl:template>',
    '  <xsl:template match="node()|@*">',
    '    <xsl:copy><xsl:apply-templates select="node()|@*"/></xsl:copy>',
    '  </xsl:template>',
    '  <xsl:output indent="yes"/>',
    '</xsl:stylesheet>',
  ].join('\n'), 'application/xml');

  const xsltProcessor = new XSLTProcessor();
  xsltProcessor.importStylesheet(xsltDoc);
  const resultDoc = xsltProcessor.transformToDocument(xmlDoc);
  return new XMLSerializer().serializeToString(resultDoc);
};

module.exports = {
  getColors: getColors,
  addRectBefore: addRectBefore,
  addShadowFilter: addShadowFilter,
  prettifyXml: prettifyXml,
  svgWithSize: svgWithSize,
  getSVGSize: getSVGSize,
  changeColorByIndex: changeColorByIndex,
  changeAllColors: changeAllColors,
};