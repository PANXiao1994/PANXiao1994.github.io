---
title: "Contact Me"
date: 2025-06-30T00:00:00-00:00
draft: false
---

If you have any questions or want to get in touch, please use the form below:

<form action="https://formspree.io/f/xwpbywrz" method="POST" class="contact-grid">
  <div>
    <label for="name">Your Name:</label>
    <input type="text" id="name" name="name" required>
  </div>

  <div>
    <label for="email">Your Email:</label>
    <input type="email" id="email" name="_replyto" required>
  </div>

  <div class="full-width">
    <label for="message">Message:</label>
    <textarea id="message" name="message" rows="5" required></textarea>
  </div>

  <div class="full-width">
    <button type="submit">Send</button>
  </div>
</form>

<style>
.contact-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 1rem;
  max-width: 600px;
  margin: 2rem 0;
}
.contact-grid div {
  display: flex;
  flex-direction: column;
}
.contact-grid label {
  font-weight: bold;
  margin-bottom: 0.25rem;
}
.contact-grid input,
.contact-grid textarea {
  padding: 0.5rem;
  font-size: 1rem;
  border: 1px solid #ccc;
  border-radius: 4px;
}
.contact-grid .full-width {
  grid-column: 1 / -1; /* span both columns */
}
.contact-grid button {
  padding: 0.75rem;
  font-size: 1rem;
  background: #333;
  color: #fff;
  border: none;
  border-radius: 4px;
  cursor: pointer;
}
.contact-grid button:hover {
  background: #555;
}
@media (max-width: 600px) {
  .contact-grid {
    grid-template-columns: 1fr; /* collapse to single column */
  }
}
</style>
