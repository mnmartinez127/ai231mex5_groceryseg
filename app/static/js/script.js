const IMAGE_INTERVAL_MS = 100;
let uniqToken = 0;

// Function to start image processing in the browser
// and send images to the backend via WebSocket connection
const startImageProcessing = (video, canvas, image, deviceId, sessionToken) => {
  const fpsDisplay = document.getElementById("fps");
  const latencyDisplay = document.getElementById("latency");
  const socket = new WebSocket(
    `ws://${location.host}/ws_image_processing/${sessionToken}`
  );
  let intervalId;
            time_start = performance.now();
  // Connection opened
  socket.addEventListener("open", function () {
    // Start reading video from device
    navigator.mediaDevices
      .getUserMedia({
        audio: false,
        video: {
          deviceId,
          width: { max: 640 },
          height: { max: 480 },
        },
      })
      .then(function (stream) {
        video.srcObject = stream;
        // Initialize latency timer
        time_start = performance.now();
        video.play().then(() => {
          // Adapt overlay canvas size to the video size
          canvas.width = video.videoWidth;
          canvas.height = video.videoHeight;

          // Send an image in the WebSocket every 42 ms
          intervalId = setInterval(() => {
            // Create a virtual canvas to draw current video image
            const canvas = document.createElement("canvas");
            const ctx = canvas.getContext("2d");
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            // apply image filters before sending
            ctx.filter = video.style.filter;
            ctx.drawImage(video, 0, 0);
            // convert it to JPEG and send it to the websocket
            canvas.toBlob((blob) => socket.send(blob), "image/jpeg");
          }, IMAGE_INTERVAL_MS);
        });
      });
  });

  // Listen for messages from the backend to display the image
  socket.addEventListener("message", function (event) {
    image.setAttribute("src", event.data);
    time_end = performance.now();
    if (time_end-time_start > 0.0) {
      latency = (time_end-time_start);
      fps = (1000/latency);
      fpsDisplay.textContent = "FPS: "+fps.toFixed(2);
      latencyDisplay.textContent = "Latency: "+latency.toFixed(2)+"ms";
      time_start = time_end
    };

  });

  // Close the WebSocket connection when closing the video stream
  socket.addEventListener("close", function () {
    window.clearInterval(intervalId);
    video.pause();
  });

  return socket;
};


// Define event handlers on startup
window.addEventListener("DOMContentLoaded", (event) => {
  const video = document.getElementById("video");
  const canvas = document.getElementById("canvas");
  const image = document.getElementById("processed-image");
  const cameraSelect = document.getElementById("camera-select");
  const connectForm = document.getElementById("form-connect");
  const startButton = document.getElementById("button-start")



  var conf = document.getElementById("conf")
  var iou = document.getElementById("iou")
  var zoomScale = document.getElementById("zoom-scale")
  var cx = document.getElementById("cx")
  var cy = document.getElementById("cy")
  var showBoxes = document.getElementById("show-boxes")
  var showMasks = document.getElementById("show-masks")
  var rescale = document.getElementById("rescale")


  // initialize socket with a "unique" id based on connection date
  let socket;
  uniqToken = Date.now();
  console.log("Token for this session:", uniqToken);

  // List available cameras and fill select
  navigator.mediaDevices.getUserMedia({ audio: false, video: true }).then(() => {
    navigator.mediaDevices.enumerateDevices().then((devices) => {
      for (const device of devices) {
        if (device.kind === "videoinput" && device.deviceId) {
          const deviceOption = document.createElement("option");
          deviceOption.value = device.deviceId;
          deviceOption.innerText = device.label;
          cameraSelect.appendChild(deviceOption);
        }
      }
    });
  });

  // Handler to start the video connection
  connectForm.addEventListener("submit", (event) => {
      event.preventDefault();
      // Close previous socket if there is one
      if (socket) {
        socket.close();
      }
      if (startButton.textContent == "Start"){
        startButton.textContent = "Stop";
        startButton.classList = "btn btn-danger";
        const deviceId = cameraSelect.selectedOptions[0].value;
        socket = startImageProcessing(video, canvas, image, deviceId, uniqToken);
        } else {
        startButton.textContent = "Start";
        startButton.classList = "btn btn-success";
        video
      }
    });


});
// =========================================







function applyImageFilters() {
  const brightnessValue = document.getElementById("brightness").value;
  const contrastValue = document.getElementById("contrast").value;
  const grayscaleValue = document.getElementById("grayscale").value;
  const saturateValue = document.getElementById("saturate").value;

  const filters = `brightness(${brightnessValue}%) contrast(${contrastValue}%) grayscale(${grayscaleValue}%) saturate(${saturateValue}%)`;

  // Apply the filters to the input video
  const applyVideo = document.getElementById("video");
  applyVideo.style.filter = filters;
}

function resetFilters() {
  // Set default values for each filter
  document.getElementById("brightness").value = 100;
  document.getElementById("contrast").value = 100;
  document.getElementById("grayscale").value = 0;
  document.getElementById("saturate").value = 100;

  // Apply the default filters
  applyImageFilters();
}

console.log("Show Options");
document.querySelectorAll("#display-settings input").forEach((checkbox) => {
  console.log(checkbox.name);
  console.log(checkbox.value);
  console.log(checkbox.checked);
  checkbox.addEventListener("change", submitUpdatedCheck)
});
console.log("Slider Options");
document.querySelectorAll("#detection-settings input").forEach((slider) => {
  console.log(slider.name);
  console.log(slider.value);
  slider.addEventListener("input", submitUpdatedVar)
});
console.log("Image Filters")
document.querySelectorAll("#image-filters input").forEach((slider) => {
  console.log(slider.name);
  console.log(slider.value);
  slider.addEventListener("input", applyImageFilters)
});


function submitUpdatedCheck(event) {
  const selectedOption = event.target.name;
  const selectedValue = event.target.value;
  const selectedCheck = event.target.checked;
  const apiUrl = `https://${location.host}/update_session_var/${uniqToken}/${selectedOption}/${selectedValue}=${selectedCheck}`;
  console.log("Submitting new parameter with URL: ",apiUrl)
  const headers = new Headers();
  headers.append("X-Session-Token", uniqToken);

  fetch(apiUrl, {
    method: "PUT",
    headers: headers,
  })
    .then((response) => response.json())
    .then((data) => {
      console.log("Response:", data);
    })
    .catch((error) => {
      console.error("Error:", error);
    });
}



function submitUpdatedVar(event) {
  const selectedOption = event.target.name;
  const selectedValue = event.target.value;
  const apiUrl = `https://${location.host}/update_session_var/${uniqToken}/${selectedOption}/${selectedValue}`;
  console.log("Submitting new parameter with URL: ",apiUrl)
  const headers = new Headers();
  headers.append("X-Session-Token", uniqToken);

  fetch(apiUrl, {
    method: "PUT",
    headers: headers,
  })
    .then((response) => response.json())
    .then((data) => {
      console.log("Response:", data);
    })
    .catch((error) => {
      console.error("Error:", error);
    });
}
