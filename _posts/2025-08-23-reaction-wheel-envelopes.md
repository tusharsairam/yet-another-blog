---
layout: post
title:  "Reaction Wheel Envelopes"
---

Over this weekend, I stumbled upon this fairly old research paper by F. L. Markley et. al. from NASA. This paper dives into developing algorithms for calculating what are called torque and momentum envelopes for reaction wheel arrays. I figured I could spend some time reading through this paper and implementing the envelope computation in Python.

An envelope is a graph (often a 3D plot) that showcases the torque/angular momentum capabilities offered by an array of wheels. Typically in much of the spacecraft we know, a minimum of 3 reaction wheels are used. One to provide torque along one axis. This is sufficient to provide precise attitude control… until one of the wheels break down. Reaction wheels are unfortunately highly prone to failure and the spacecraft loses a degree of freedom when one decides to throw in the towel. This can jeopardize the science mission altogether. To reduce the chances of such failure, engineers bring a fourth wheel into the playground and rearrange the array of wheels. This introduces redundancy, so if one wheel gives up, the other three wheels will continue to provide 3-axis control. Some spacecraft even go for more, such as the Swift Gamma Ray Burst Explorer that uses 6 wheels.

Envelopes are useful to assess how helpful can a reaction wheel array be in terms of torque and/or angular momentum storage capabilities. Being able to develop these also helps design strategies for wheel desaturation, tells you how much torque authority is provided, and also gives you a picture of its limits. Great for sizing your wheels and planning movements that are within the prescribed limits of the array.

Here is an example of the torque envelope I generated considering a simple 3-wheel array, where the spin axes of the wheels are orthogonal to each other and are aligned with the spacecraft axes. 
