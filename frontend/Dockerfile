FROM node:18

WORKDIR /app

COPY package.json ./
COPY tailwind.config.js ./
COPY postcss.config.js ./
COPY public ./public
COPY src ./src

RUN npm install
RUN npm run build

EXPOSE 3000
CMD ["npm", "start"]
