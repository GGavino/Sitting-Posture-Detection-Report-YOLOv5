const express = require('express');
const multer = require('multer');
const axios = require('axios');
const cors = require('cors');
const path = require('path');
const fs = require('fs');
const FormData = require('form-data'); // <-- Add this

const app = express();
app.use(cors());
const upload = multer({ dest: 'uploads/' });

app.use(express.static('public'));

app.post('/upload', upload.single('video'), async (req, res) => {
    try {
        const video = req.file;
        const formData = new FormData();
        formData.append('file', fs.createReadStream(video.path), video.originalname);

        const response = await axios.post(
            'http://localhost:8000/analyze/',
            formData,
            { headers: formData.getHeaders() }
        );
        res.json(response.data);
    } catch (err) {
        res.status(500).json({ error: err.toString() });
    }
});

app.get('/download/csv', async (req, res) => {
    const response = await axios.get('http://localhost:8000/download/csv', { responseType: 'stream' });
    res.setHeader('Content-Disposition', 'attachment; filename=posture_report.csv');
    response.data.pipe(res);
});

app.listen(3000, () => console.log('Frontend server running on http://localhost:3000'));