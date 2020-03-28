import p5 from "p5";
import * as tf from "@tensorflow/tfjs";
import { model, labelList } from "./train";

let r, g, b;
let w;
let factor = 0.3;

const setup = p5 => {
	const q = document.querySelector("#layers");
	for (let i = 0; i < 1; i++) {
		let p = p5.createElement(
			"model-element",
			`<div class="model-element"><h2>Dense (16, sigmoid)<h2></div>`
		);
		p.id(`e-${i}`);
		p.parent(q);
	}

	if (p5.windowWidth >= 1100) factor = 0.4;
	else if (p5.windowWidth >= 670 && p5.windowWidth < 1100) factor = 0.5;
	else factor = 0.7;

	w = p5.windowWidth * factor;
	const c = p5.createCanvas(w, w);
	c.parent("sketch-holder");
	r = p5.random(255);
	g = p5.random(255);
	b = p5.random(255);
	document.querySelector("#e-i").style.backgroundColor = p5.color(r, g, b);
	document.querySelector(".r").style.backgroundColor = p5.color(r, 0, 0);
	document.querySelector(".g").style.backgroundColor = p5.color(0, g, 0);
	document.querySelector(".b").style.backgroundColor = p5.color(0, 0, b);
	let r_percent = p5.floor((r / 255) * 100);
	let g_percent = p5.floor((g / 255) * 100);
	let b_percent = p5.floor((b / 255) * 100);
	document.querySelector(
		".r"
	).style.backgroundImage = `linear-gradient(to right, transparent ${r_percent}%, white 0)`;
	document.querySelector(
		".g"
	).style.backgroundImage = `linear-gradient(to right, transparent ${g_percent}%, white 0)`;
	document.querySelector(
		".b"
	).style.backgroundImage = `linear-gradient(to right, transparent ${b_percent}%, white 0)`;
};

const windowResized = p5 => {
	if (p5.windowWidth >= 1100) factor = 0.4;
	else if (p5.windowWidth >= 670 && p5.windowWidth < 1100) factor = 0.5;
	else factor = 0.7;
	w = p5.windowWidth * factor;
	p5.resizeCanvas(w, w);
};

const draw = p5 => {
	p5.background(r, g, b);
	document.querySelector("#new").addEventListener("click", () => {
		r = p5.random(255);
		g = p5.random(255);
		b = p5.random(255);
		document.querySelector("#e-i").style.backgroundColor = p5.color(r, g, b);
		document.querySelector(".r").style.backgroundColor = p5.color(r, 0, 0);
		document.querySelector(".g").style.backgroundColor = p5.color(0, g, 0);
		document.querySelector(".b").style.backgroundColor = p5.color(0, 0, b);
		let r_percent = (r / 255) * 100;
		let g_percent = (g / 255) * 100;
		let b_percent = (b / 255) * 100;
		document.querySelector(
			".r"
		).style.backgroundImage = `linear-gradient(to right, transparent ${r_percent}%, white 0)`;
		document.querySelector(
			".g"
		).style.backgroundImage = `linear-gradient(to right, transparent ${g_percent}%, white 0)`;
		document.querySelector(
			".b"
		).style.backgroundImage = `linear-gradient(to right, transparent ${b_percent}%, white 0)`;
	});
};

const sketch = p5 => {
	p5.setup = () => setup(p5);
	p5.draw = () => draw(p5);
	p5.windowResized = () => windowResized(p5);
};

new p5(sketch);

document.querySelector("#predict").addEventListener("click", () => {
	predict();
});

function predict() {
	let label;
	if (r === g && g === b) {
		if (r <= 30 && g <= 30 && b <= 30) {
			label = "black";
		} else if (r === 255 && g === 255 && b === 255) {
			label = "white";
		} else {
			label = "grey-ish";
		}
	} else {
		tf.tidy(() => {
			const x = tf.tensor2d([[r / 255, g / 255, b / 255]]);
			let results = model.predict(x);
			let index = results.argMax(1).dataSync()[0];
			label = labelList[index];
		});
	}
	document.querySelector("#prediction").innerHTML = label;
	document.querySelector(
		"#e-o"
	).innerHTML = `<h2>Dense (9, softmax) → ${label}</h2>`;
}
