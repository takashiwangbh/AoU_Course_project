<table style="border-collapse: collapse; width: 100%; text-align: center; font-family: Arial, sans-serif;">
    <tr style="background-color: #007BFF; color: white;">
        <th style="padding: 10px; border: 1px solid white;">Signal Processing Method</th>
        <th style="padding: 10px; border: 1px solid white;">Description</th>
        <th style="padding: 10px; border: 1px solid white;">SVM Accuracy</th>
        <th style="padding: 10px; border: 1px solid white;">Observations</th>
    </tr>
    <tr style="background-color: #E3F2FD;">
        <td style="padding: 10px; border: 1px solid #007BFF;"> <b>Original Data</b></td>
        <td style="padding: 10px; border: 1px solid #007BFF;">Raw hand gesture data without filtering.</td>
        <td style="padding: 10px; border: 1px solid #007BFF;"> <b>47.33%</b></td>
        <td style="padding: 10px; border: 1px solid #007BFF;">High noise, poor classification performance.</td>
    </tr>
    <tr style="background-color: #BBDEFB;">
        <td style="padding: 10px; border: 1px solid #007BFF;"><b>Moving Average Filter</b></td>
        <td style="padding: 10px; border: 1px solid #007BFF;">Simple smoothing method that averages values.</td>
        <td style="padding: 10px; border: 1px solid #007BFF;"><b>59.33%</b></td>
        <td style="padding: 10px; border: 1px solid #007BFF;">Some noise reduction, but also loss of details.</td>
    </tr>
    <tr style="background-color: #E3F2FD;">
        <td style="padding: 10px; border: 1px solid #007BFF;"> <b>Gaussian Filter</b></td>
        <td style="padding: 10px; border: 1px solid #007BFF;">Advanced smoothing using a Gaussian kernel.</td>
        <td style="padding: 10px; border: 1px solid #007BFF;"><b>69.33%</b></td>
        <td style="padding: 10px; border: 1px solid #007BFF;">Better smoothing while retaining gesture shape.</td>
    </tr>
    <tr style="background-color: #BBDEFB;">
        <td style="padding: 10px; border: 1px solid #007BFF;"><b>Kalman Filter</b></td>
        <td style="padding: 10px; border: 1px solid #007BFF;">Adaptive filtering that preserves trends.</td>
        <td style="padding: 10px; border: 1px solid #007BFF;"> <b>81.33%</b></td>
        <td style="padding: 10px; border: 1px solid #007BFF;">Best performance, reducing noise effectively.</td>
    </tr>
</table>
