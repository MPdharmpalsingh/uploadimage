var express = require('express');
var router = express.Router();
const multer = require("multer");

/* GET home page. */
router.get('/', function(req, res, next) {
  res.render('index', { title: 'Express' });
});

const upload = multer() 
router.post("/upload",upload.single("file"), function(req, res, next) {
  console.log("req.body");
});

module.exports = router;
