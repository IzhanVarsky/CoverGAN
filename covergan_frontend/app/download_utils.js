import config from './config.json';

function downloadBase64File(contentType, base64Data, fileName) {
  const link = document.createElement("a");
  link.href = `data:${contentType};base64,${base64Data}`;
  link.download = fileName;
  link.click();
}

function downloadTextFile(text, filename) {
  const link = document.createElement('a');
  link.href = 'data:text/plain;charset=utf-8,' + encodeURIComponent(text);
  link.download = filename;
  link.click();
}

function downloadPNGFromServer(data) {
  const formData = new FormData()
  formData.append("svg", data);
  $.ajax({
    url: `${config.host}/rasterize`,
    type: 'POST',
    data: formData,
    processData: false,
    contentType: false,
    cache: false,
    success: (response) => {
      console.log('SUCC', response);
      downloadBase64File("image/png", response.result.res_png1, "rasterized.png");
    },
    error: (e) => {
      console.log('ERR', e);
    }
  });
}

function getJSON(data, callback) {
  const formData = new FormData()
  formData.append("svg", data);
  $.ajax({
    url: `${config.host}/svg_to_json`,
    type: 'POST',
    data: formData,
    processData: false,
    contentType: false,
    cache: false,
    success: (response) => {
      console.log('SUCC', response);

      if (callback) {
        callback(response.result);
      } else {
        downloadTextFile(JSON.stringify(response.result), "obj.json");
      }
    },
    error: (e) => {
      console.log('ERR', e);
    }
  });
}

function extractColors(image, n, callback, callback_err) {
  const formData = new FormData()
  formData.append("img", image);
  formData.append("color_count", n);
  formData.append("algo_type", 1);
  formData.append("use_random", false);
  $.ajax({
    url: `${config.host}/extract_colors`,
    type: 'POST',
    data: formData,
    processData: false,
    contentType: false,
    cache: false,
    success: (response) => {
      console.log('COLORS:', response);
      const colors = response.result.map(([r, g, b]) => `rgb(${r}, ${g}, ${b})`)
      callback(colors);
    },
    error: (e) => {
      if (callback_err !== undefined) {
        callback_err(e)
      }
      console.log('ERR', e);
    }
  });
}

module.exports = {
  downloadBase64File: downloadBase64File,
  downloadTextFile: downloadTextFile,
  getJSON: getJSON,
  extractColors: extractColors,
  downloadPNGFromServer: downloadPNGFromServer,
};