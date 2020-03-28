import * as tf from "@tensorflow/tfjs";
import * as tfvis from "@tensorflow/tfjs-vis";
import colorData from "./data/colorData.json";

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

data = colorData;
shortData = { entries: [] };
for (let i = 0; i < 25000; i++) {
	shortData.entries.push(data.entries[i]);
}
prepare(shortData);

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
		document.querySelector("#load").setAttribute("disabled", "true");
		document.querySelector("#predict").attributes.removeNamedItem("disabled");
	})
);

document.querySelector("#load").addEventListener("click", async () => {
	model = await tf.loadLayersModel(
		"https://raw.githubusercontent.com/VarunIrani/color-classifier/master/src/model/color-model.json"
	);
	document.querySelector("#train").setAttribute("disabled", "true");
	document.querySelector("#predict").attributes.removeNamedItem("disabled");
	document.querySelector("#load").setAttribute("disabled", "true");
});

export { model, labelList };
