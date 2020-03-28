// import "./modelElement.js";
let data;
let model;
let xs, ys;
let numEpochs = 100;
let labelList = [
	"red-ish",
	"green-ish",
	"blue-ish",
	"orange-ish",
	"yellow-ish",
	"pink-ish",
	"purple-ish",
	"brown-ish",
	"grey-ish"
];
let colors = [];
let labels = [];
let shortData;

$.getJSON("./js/colorData.json", res => {
	data = res;
	shortData = { entries: [] };
	for (let i = 0; i < 25000; i++) {
		shortData.entries.push(data.entries[i]);
	}
	prepare(shortData);
});

function prepare(data) {
	for (let record of data.entries) {
		let col = [record.r / 255, record.g / 255, record.b / 255];
		colors.push(col);
		labels.push(labelList.indexOf(record.label));
	}
	xs = tf.tensor2d(colors);
	let labelsTensor = tf.tensor1d(labels, "int32");
	ys = tf.oneHot(labelsTensor, 9);
	labelsTensor.dispose();

	model = tf.sequential();

	model.add(
		tf.layers.dense({
			units: 16,
			activation: "sigmoid",
			inputShape: [3]
		})
	);

	model.add(
		tf.layers.dense({
			units: ys.shape[1],
			activation: "softmax"
		})
	);

	const optimizer = tf.train.adam(0.01);
	model.compile({
		optimizer: optimizer,
		loss: "categoricalCrossentropy",
		metrics: ["accuracy"]
	});
}

async function train(model, xs, ys, fitCallbacks) {
	const options = {
		epochs: numEpochs,
		validationSplit: 0.2,
		callbacks: fitCallbacks,
		batchSize: Math.floor(ys.shape[0] / 5),
		shuffle: true
	};
	return await model.fit(xs, ys, options);
}

async function watchTraining() {
	const metrics = ["loss", "val_loss", "acc", "val_acc"];
	const container = {
		name: "Model Accuracy",
		tab: "Training",
		styles: {
			height: "1000px"
		}
	};
	const callbacks = tfvis.show.fitCallbacks(container, metrics);
	return await train(model, xs, ys, callbacks);
}

document.querySelector("#train").addEventListener("click", () =>
	watchTraining().then(() => {
		setTimeout(() => tfvis.visor().toggle(), 3000);
		xs.dispose();
		ys.dispose();
		document.querySelector("#train").setAttribute("disabled", "true");
		document.querySelector("#predict").attributes.removeNamedItem("disabled");
	})
);

let r, g, b;
let w;
let factor = 0.3;

function setup() {
	const q = document.querySelector("#layers");
	for (let i = 0; i < 1; i++) {
		let p = createElement(
			"model-element",
			`<div class="model-element"><h2>Dense (16, sigmoid)<h2></div>`
		);
		p.id(`e-${i}`);
		p.parent(q);
	}

	if (windowWidth >= 1100) factor = 0.4;
	else if (windowWidth >= 670 && windowWidth < 1100) factor = 0.5;
	else factor = 0.7;

	w = windowWidth * factor;
	const c = createCanvas(w, w);
	c.parent("sketch-holder");
	r = random(255);
	g = random(255);
	b = random(255);
	document.querySelector("#e-i").style.backgroundColor = color(r, g, b);
	document.querySelector(".r").style.backgroundColor = color(r, 0, 0);
	document.querySelector(".g").style.backgroundColor = color(0, g, 0);
	document.querySelector(".b").style.backgroundColor = color(0, 0, b);
	let r_percent = floor((r / 255) * 100);
	let g_percent = floor((g / 255) * 100);
	let b_percent = floor((b / 255) * 100);
	document.querySelector(
		".r"
	).style.backgroundImage = `linear-gradient(to right, transparent ${r_percent}%, white 0)`;
	document.querySelector(
		".g"
	).style.backgroundImage = `linear-gradient(to right, transparent ${g_percent}%, white 0)`;
	document.querySelector(
		".b"
	).style.backgroundImage = `linear-gradient(to right, transparent ${b_percent}%, white 0)`;
}

function windowResized() {
	if (windowWidth >= 1100) factor = 0.4;
	else if (windowWidth >= 670 && windowWidth < 1100) factor = 0.5;
	else factor = 0.7;
	w = windowWidth * factor;
	resizeCanvas(w, w);
}

function draw() {
	background(r, g, b);
	document.querySelector("#new").addEventListener("click", () => {
		r = random(255);
		g = random(255);
		b = random(255);
		document.querySelector("#e-i").style.backgroundColor = color(r, g, b);
		document.querySelector(".r").style.backgroundColor = color(r, 0, 0);
		document.querySelector(".g").style.backgroundColor = color(0, g, 0);
		document.querySelector(".b").style.backgroundColor = color(0, 0, b);
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
}

document.querySelector("#predict").addEventListener("click", () => {
	predict();
});

function predict() {
	let label;
	if (r == g && g == b) {
		if (r <= 30 && g <= 30 && b <= 30) {
			label = "black";
		} else if (r == 255 && g == 255 && b == 255) {
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

document.querySelector("#load").addEventListener("click", async () => {
	model = await tf.loadLayersModel("./model/color-model.json");
	document.querySelector("#train").setAttribute("disabled", "true");
	document.querySelector("#predict").attributes.removeNamedItem("disabled");
	document.querySelector("#load").setAttribute("disabled", "true");
});
