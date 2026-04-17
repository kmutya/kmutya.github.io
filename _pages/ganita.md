---
title: "Posts"
permalink: /ganita/
author_profile: true
classes: wide
---

<ul style="list-style:none; padding:0; margin:0;">
{% for post in site.posts %}
  <li style="font-size:0.8em; margin:0.4em 0;"><a href="{{ post.url | relative_url }}" style="text-decoration:none;">{{ post.title }}</a> &nbsp;<span style="color:#aaa;">{{ post.date | date: "%b %Y" }}</span></li>
{% endfor %}
</ul>
