let screenwidth = 500;
let screenheight = 500;
function preload() {
  // img = loadImage('assets/clockwise.png');
}
function setup() {
  let cnv = createCanvas(screenwidth, screenheight);
  cnv.parent('container');
}

function draw() {}
const centerX = 250;
const length = 125;
const width = 25;
function update(data, info) {
  clear();
  background(240);
  let last_u = info.last_u;
  let episode = info.episode;
  let state = JSON.parse(data);

  let theta = state[0];

  let axlePos = [centerX, 250];

  // pole
  let dx = Math.sin(theta) * length;
  let dy = Math.cos(theta) * length;
  let linePos = [axlePos[0], axlePos[1], axlePos[0] + dx, axlePos[1] - dy];
  stroke(204, 76, 76);
  strokeWeight(width);
  line(...linePos);

  // axle
  noStroke()
  fill(51, 56, 68);
  circle(axlePos[0], axlePos[1], width / 2);

  // fill(51, 56, 64);
  // noStroke();
  // textSize(18);
  // if (rewards !== undefined) {
  //   text(`Return ${rewards}`, 14, 60);
  // }
  if (episode !== undefined) {
    text(`Episode ${episode}`, 14, 30);
  }
}
