<?php
extract($_REQUEST);
require_once('voicerss_tts.php');
?>
<html>
<head>
<title>Untitled Document</title>
</head>

<body>
<?php
$name=$mess;
if($name!="")
{
$message=$name;

$tts = new VoiceRSS;
$voice = $tts->speech(array(
    'key' => '84690531ea3147658ee11c95c69ed82f',
    'hl' => 'en-us',
    'src' => "$message",
    'r' => '0',
    'c' => 'mp3',
    'f' => '44khz_16bit_stereo',
    'ssml' => 'false',
    'b64' => 'true'
));

//echo '<audio src="' . $voice['response'] . '" autoplay="autoplay"></audio>';
?>
<embed src="<?php echo $voice['response']; ?>" autostart="true" width="50" height="20"></embed>

<?php
}
?>
<script>
						setTimeout(function () {
						   //Redirect with JavaScript
						  window.location.href="http://localhost:5000/clear_data";
						}, 10000);
						</script>
</body>
</html>
