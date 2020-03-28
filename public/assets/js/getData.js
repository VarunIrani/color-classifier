const config = {
	apiKey: "AIzaSyDPekCKX4ee6h9NVR2lEITGAM0XIHn-c7c",
	authDomain: "color-classification.firebaseapp.com",
	databaseURL: "https://color-classification.firebaseio.com",
	projectId: "color-classification",
	storageBucket: "",
	messagingSenderId: "590040209608"
};
firebase.initializeApp(config);
database = firebase.database();

const colorsRef = database.ref("colors");

const filter = {
	YGdqOTDDfpqsSD6CvNNFQmRp9sJDdI1QJm32: true,
	HUXmyv1dSSUnIvYk976MPWUSaTG2: true,
	hPdk0Qpo0Gb5NsWSgxsqPM7M2EA2: true
};

colorsRef.once("value").then(results => {
	let data = results.val();
	let keys = Object.keys(data);

	let allData = { entries: [] };

	for (let key of keys) {
		let record = data[key];
		let id = record.uid;
		if (!filter[id]) allData.entries.push(record);
	}

	download(JSON.stringify(allData), "colorData.json", "text/plain");
});

function download(content, fileName, contentType) {
	var a = document.createElement("a");
	var file = new Blob([content], { type: contentType });
	a.href = URL.createObjectURL(file);
	a.download = fileName;
	a.click();
}
