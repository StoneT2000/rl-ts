let screenwidth = 400;
let screenheight = 400;
function setup() {
  let cnv = createCanvas(screenwidth, screenheight);
  cnv.parent('container');
}

function draw() {}
const centerX = 200;
const polewidth = 10;
const cartwidth = 50;
const cartheight = 30;
const carty = 100;
function update(data, info) {
  clear();
  background(240);
  let x_threshold = info.x_threshold;

  let world_width = x_threshold * 2;
  scale = screenwidth / world_width;
  let poleLength = info.pole_length * scale * 2;

  let rewards = info.rewards;
  let episode = info.episode;

  let state = JSON.parse(data);

  let x = state[0];
  let theta = state[2];

  let axlePos = [x * scale + centerX, 340];

  // background line
  strokeWeight(2);
  stroke(170, 170, 170);
  line(0, axlePos[1], screenwidth, axlePos[1]);

  // cart
  noStroke();
  fill(51, 56, 68);
  rect(axlePos[0] - cartwidth / 2, axlePos[1] - cartheight / 4, cartwidth, cartheight);

  // axle
  fill(127, 127, 204);
  circle(axlePos[0], axlePos[1], polewidth);

  // pole
  let dx = Math.sin(theta) * poleLength;
  let dy = Math.cos(theta) * poleLength;
  let linePos = [axlePos[0], axlePos[1], axlePos[0] + dx, axlePos[1] - dy];
  strokeCap(SQUARE);
  stroke(204, 153, 102);
  strokeWeight(polewidth);
  line(...linePos);

  fill(51, 56, 68);
  noStroke();
  textSize(18);
  if (rewards !== undefined) {
    text(`Return ${rewards}`, 14, 60);
  }
  if (episode !== undefined) {
    text(`Episode ${episode}`, 14, 30);
  }
}
